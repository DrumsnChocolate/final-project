import argparse
import time
from typing import Callable, Any, List

import numpy as np
import scipy
import torch
import yaml
from prodict import Prodict
from torch.optim import SGD
from tqdm import tqdm
from configs.config_options import DictAction
from configs.config_validation import validate_cfg
from finetune.checkpoint import checkpoint
from finetune.loss import build_loss_function, call_loss
from logger import IterationLogger, EpochLogger
from metrics import call_metrics, build_metric_functions, append_metrics, average_metrics
from models import build_model, SamWrapper
from datasets.loaders import build_dataloaders
import os.path as osp


class InfiniteIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the segment anything model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
    )
    return parser.parse_args()


def load_cfg(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def override_cfg(cfg, cfg_overrides):
    for k, v in cfg_overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k, None), dict):
            cfg[k] = override_cfg(cfg[k], v)
        else:
            cfg[k] = v
    return cfg


def load_cfg_from_file(filename):
    cfg = load_cfg(filename)
    bases = cfg.get('_bases_', None)
    if bases is not None:
        del cfg['_bases_']
        for base in bases:
            cfg = override_cfg(load_cfg(base), cfg)
    return cfg


def get_cfg_dict(args):
    cfg = load_cfg_from_file(args.config)
    if args.cfg_options is not None:
        cfg = override_cfg(cfg, args.cfg_options)
    if cfg.get('out_dir') is None:
        cfg['out_dir'] = 'outputs'
    return cfg


def cfg_to_prodict(cfg):
    cfg = Prodict.from_dict(cfg)
    cfg.data.preprocess = [Prodict.from_dict(step) for step in cfg.data.preprocess]
    cfg.model.loss.parts = [Prodict.from_dict(loss_item) for loss_item in cfg.model.loss.parts]
    cfg.model.metrics = [Prodict.from_dict(metric) for metric in cfg.model.metrics]
    return cfg


def get_cfg(args):
    cfg = cfg_to_prodict(get_cfg_dict(args))
    validate_cfg(cfg)
    return cfg


def dump_cfg(cfg):
    return yaml.dump(cfg.to_dict(is_recursive=True, exclude_none=True, exclude_none_in_lists=True))


def store_cfg(cfg, logger):
    with open(osp.join(logger.log_dir, 'config.yaml'), 'w') as f:
        f.write(dump_cfg(cfg))


def get_logger(cfg):
    logger = None
    if cfg.schedule.iterations is not None:
        logger = IterationLogger(cfg)
    elif cfg.schedule.epochs is not None:
        logger = EpochLogger(cfg)
    if logger is None:
        raise NotImplementedError()
    logger.log(dump_cfg(cfg), to_file=False)
    return logger


def build_optimizer(cfg, model):
    if cfg.model.optimizer.name == 'sgd':
        return SGD(
            model.parameters(),
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.wd,
            momentum=cfg.model.optimizer.momentum
        )
    raise NotImplementedError()


def get_foreground_points(targets):
    foreground_points = []

    for target in targets:
        # we assume there's only one target
        assert target.shape[0] == 1
        distance_transform = scipy.ndimage.distance_transform_edt(target[0].cpu())
        foreground_point = np.argmax(distance_transform)
        foreground_point = np.unravel_index(foreground_point, distance_transform.shape)
        foreground_points.append(foreground_point)
    return torch.Tensor(foreground_points).to(targets.device).unsqueeze(1)


def get_random_foreground_points(targets):
    foreground_points = []
    for target in targets:
        # we assume there's only one target
        assert target.shape[0] == 1
        xs, ys = np.where(target[0].cpu() == 1.0)
        index = np.random.choice(np.arange(len(xs)))
        foreground_point = xs[index], ys[index]
        foreground_points.append(foreground_point)
    return torch.Tensor(foreground_points).to(targets.device).unsqueeze(1)


def train_epoch(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, dataloaders, epoch, logger: EpochLogger):
    train_loader = dataloaders['train']
    model.train()
    total_epoch_train_loss = 0
    for i, batch in tqdm(enumerate(train_loader)):
        samples, targets, classes = batch
        foreground_points = get_random_foreground_points(targets)
        outputs = model(samples, foreground_points)
        loss = call_loss(loss_function, outputs, targets, cfg)
        metrics = call_metrics(metric_functions, outputs, targets, model)
        assert metrics.get('loss') is None
        metrics['loss'] = loss.tolist()
        logger.log_batch_metrics(metrics)
        total_epoch_train_loss += loss
        loss.backward()
        optimizer.step()
    logger.log_epoch(epoch)


def train_iteration(cfg, model: SamWrapper, loss_function: Callable, metric_functions: dict[str, Callable], optimizer,
                    dataloaders, iteration, logger: IterationLogger):
    infinite_train_loader = dataloaders['infinite_train']
    model.train()
    batch = next(infinite_train_loader)
    samples, targets, classes = batch
    foreground_points = get_random_foreground_points(targets)
    outputs = model(samples, foreground_points)
    # print(outputs)
    loss = call_loss(loss_function, outputs, targets, cfg)
    metrics = call_metrics(metric_functions, outputs, targets, model)
    assert metrics.get('loss') is None
    metrics['loss'] = loss.tolist()
    logger.log_iteration_metrics(metrics, iteration)
    loss.backward()
    optimizer.step()


def validate_epoch(cfg, model: SamWrapper, loss_function, metric_functions, dataloaders, logger: EpochLogger):
    logger.log('Validation')
    val_loader = dataloaders['val']
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            samples, targets, classes = batch
            foreground_points = get_foreground_points(targets)
            outputs = model(samples, foreground_points)
            loss = call_loss(loss_function, outputs, targets, cfg)
            metrics = call_metrics(metric_functions, outputs, targets, model)
            assert metrics.get('loss') is None
            metrics['loss'] = loss.tolist()
            logger.log_batch_metrics(metrics)
            total_val_loss += loss
    logger.log_epoch(1, split='val')


def test_epoch(cfg, model: SamWrapper, loss_function, metric_functions, dataloaders, logger: EpochLogger):
    logger.log('Testing')
    test_loader = dataloaders['test']
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            samples, targets, classes = batch
            foreground_points = get_foreground_points(targets)
            print(foreground_points.shape)
            print(foreground_points)
            outputs = model(samples, foreground_points)
            loss = call_loss(loss_function, outputs, targets, cfg)
            metrics = call_metrics(metric_functions, outputs, targets, model)
            assert metrics.get('loss') is None
            metrics['loss'] = loss.tolist()
            logger.log_batch_metrics(metrics)
            total_test_loss += loss
    logger.log_epoch(1, split='test')


def train_epochs(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, dataloaders, logger):
    for epoch in range(1, cfg.schedule.epochs + 1):
        train_epoch(cfg, model, loss_function, metric_functions, optimizer, dataloaders, epoch, logger)
        if epoch % cfg.schedule.val_interval != 0:
            continue
        validate_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)
    # test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def train_iterations(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, dataloaders, logger):
    dataloaders['infinite_train'] = InfiniteIterator(dataloaders['train'])
    for iteration in range(1, cfg.schedule.iterations + 1):
        train_iteration(cfg, model, loss_function, metric_functions, optimizer, dataloaders, iteration, logger)
        if iteration % cfg.schedule.val_interval != 0:
            continue
        validate_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)
    # test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def train(cfg):
    logger = get_logger(cfg)
    store_cfg(cfg, logger)
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg, logger)
    optimizer = build_optimizer(cfg, model)
    loss_function = build_loss_function(cfg)
    metric_functions = build_metric_functions(cfg)
    logger.log('Training')
    if cfg.schedule.iterations is not None:
        train_iterations(cfg, model, loss_function, metric_functions, optimizer, dataloaders, logger)
    elif cfg.schedule.epochs is not None:
        train_epochs(cfg, model, loss_function, metric_functions, optimizer, dataloaders, logger)
    checkpoint(cfg, model, optimizer)


def main():
    # parse config
    args = parse_args()
    cfg = get_cfg(args)
    with torch.autograd.detect_anomaly(check_nan=True):
        train(cfg)


if __name__ == '__main__':
    main()
