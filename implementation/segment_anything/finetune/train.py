import argparse
import time
from typing import Callable, Any

import numpy as np
import scipy
import torch
from prodict import Prodict
from torch.optim import SGD
from yaml import load, Loader
from configs.config_options import DictAction
from configs.config_validation import validate_cfg
from finetune.loss import build_loss_function, call_loss
from logger import Logger
from metrics import call_metrics, build_metric_functions, append_metrics, average_metrics
from models import build_model, SamWrapper
from datasets.loaders import build_dataloaders
import os.path as osp

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
        return load(f, Loader=Loader)


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
    if bases:
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


def get_logger(cfg):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    logger = Logger(log_dir=osp.join(cfg.out_dir, timestamp))
    logger.log(cfg)
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
    return torch.Tensor(foreground_points).to(targets.device).unsqueeze(0)

def get_random_foreground_points(targets):
    foreground_points = []
    for target in targets:
        # we assume there's only one target
        assert target.shape[0] == 1
        xs, ys = np.where(target[0].cpu() == 1.0)
        index = np.random.choice(np.arange(len(xs)))
        foreground_point = xs[index], ys[index]
        foreground_points.append(foreground_point)
    return torch.Tensor(foreground_points).to(targets.device).unsqueeze(0)

def train_epoch(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, dataloaders, epoch, logger):
    train_loader = dataloaders['train']
    model.train()
    total_epoch_train_loss = 0
    epoch_train_metrics = {}
    total_epoch_train_samples = 0
    for i, batch in enumerate(train_loader):
        samples, targets, classes = batch
        foreground_points = get_random_foreground_points(targets)
        outputs = model(samples, foreground_points)
        loss = call_loss(loss_function, outputs, targets, cfg)
        metrics = call_metrics(metric_functions, outputs, targets)
        assert metrics.get('loss') is None
        metrics['loss'] = loss.tolist()
        append_metrics(epoch_train_metrics, metrics)
        total_epoch_train_loss += loss
        total_epoch_train_samples += len(samples)
        loss.backward()
        optimizer.step()
    average_metrics(epoch_train_metrics)
    epoch_train_metrics['epoch'] = epoch
    logger.log_dict(epoch_train_metrics)
    logger.log(f'Epoch {epoch}, average train loss {epoch_train_metrics["avg_loss"]}')


def train_iteration(cfg, model: SamWrapper, loss_function: Callable, metric_functions: dict[str, Callable], optimizer, dataloaders, iteration, logger):
    infinite_train_loader = dataloaders['infinite_train']
    model.train()
    batch = next(infinite_train_loader)
    samples, targets, classes = batch
    foreground_points = get_random_foreground_points(targets)
    outputs = model(samples, foreground_points)
    loss = call_loss(loss_function, outputs, targets, cfg)
    metrics = call_metrics(metric_functions, outputs, targets)
    assert metrics.get('loss') is None
    metrics['loss'] = loss.tolist()
    metrics['iteration'] = iteration
    logger.log_dict(metrics)
    loss.backward()
    optimizer.step()
    if (iteration+1) % cfg.schedule.log_interval != 0:
        return
    logger.log(f'Iteration {iteration}, train loss {metrics["loss"]}')


def validate_epoch(cfg, model: SamWrapper, loss_function, metric_functions, dataloaders, logger):
    val_loader = dataloaders['val']
    model.eval()
    total_val_loss = 0
    val_metrics = {}
    total_val_samples = 0
    for i, batch in enumerate(val_loader):
        samples, targets, classes = batch
        foreground_points = get_foreground_points(targets)
        outputs = model(samples, foreground_points)
        loss = call_loss(loss_function, outputs, targets, cfg)
        metrics = call_metrics(metric_functions, outputs, targets)
        assert metrics.get('loss') is None
        metrics['loss'] = loss.tolist()
        append_metrics(val_metrics, metrics)
        total_val_loss += loss
        total_val_samples += len(samples)
    average_metrics(val_metrics)
    logger.log_dict(val_metrics)
    logger.log(f'Validation, average val loss {val_metrics["avg_loss"]}')


def test_epoch(cfg, model: SamWrapper, loss_function, metric_functions, dataloaders, logger):
    test_loader = dataloaders['test']
    model.eval()
    total_test_loss = 0
    test_metrics = {}
    total_test_samples = 0
    for i, batch in enumerate(test_loader):
        samples, targets, classes = batch
        foreground_points = get_foreground_points(targets)
        outputs = model(samples, foreground_points)
        loss = call_loss(loss_function, outputs, targets, cfg)
        metrics = call_metrics(metric_functions, outputs, targets)
        assert metrics.get('loss') is None
        metrics['loss'] = loss.tolist()
        append_metrics(test_metrics, metrics)
        total_test_loss += loss
        print(loss)
        total_test_samples += len(samples)
        break
    average_metrics(test_metrics)
    logger.log_dict(test_metrics)
    logger.log(f'Test, average test loss {test_metrics["avg_loss"]}')


def train_epochs(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, dataloaders, logger):
    for epoch in range(cfg.schedule.epochs):
        train_epoch(cfg, model, loss_function, metric_functions, optimizer, dataloaders, epoch, logger)
        if (epoch+1) % cfg.schedule.val_interval != 0:
            continue
        validate_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)
    test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


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


def train_iterations(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, dataloaders, logger):
    dataloaders['infinite_train'] = InfiniteIterator(dataloaders['train'])
    for iteration in range(cfg.schedule.iterations):
        train_iteration(cfg, model, loss_function, metric_functions, optimizer, dataloaders, iteration, logger)

        if (iteration+1) % cfg.schedule.val_interval != 0:
            continue
        validate_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)
    test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def train(cfg):
    logger = get_logger(cfg) # todo: possibly make logging more efficient? idk
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg, logger)
    optimizer = build_optimizer(cfg, model)
    loss_function = build_loss_function(cfg)
    metric_functions = build_metric_functions(cfg)
    if cfg.schedule.iterations is not None:
        train_iterations(cfg, model, loss_function, metric_functions, optimizer, dataloaders, logger)
    elif cfg.schedule.epochs is not None:
        train_epochs(cfg, model, loss_function, metric_functions, optimizer, dataloaders, logger)




def main():
    # parse config
    args = parse_args()
    cfg = get_cfg(args)
    train(cfg)


if __name__ == '__main__':
    main()
