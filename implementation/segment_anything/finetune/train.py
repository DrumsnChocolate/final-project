import argparse
import os.path as osp
import random
from typing import Callable

import numpy as np
import scipy
import torch
import yaml
from prodict import Prodict
from torch.optim import SGD
from tqdm import tqdm

from finetune.checkpoint import checkpoint
from finetune.configs.config_options import DictAction
from finetune.configs.config_validation import validate_cfg_train
from finetune.datasets.loaders import build_dataloaders
from finetune.logger import IterationLogger, EpochLogger
from finetune.loss import build_loss_function, call_loss
from finetune.metrics import call_metrics, build_metric_functions
from finetune.models.build_model import build_model
from finetune.models.sam_wrapper import SamWrapper
from finetune.scheduler import ReduceLROnPlateauScheduler, DummyScheduler
from finetune.stopper import EarlyStopper, DummyStopper, Stopper
from segment_anything.modeling import get_param_count, get_param_count_trainable_recursively


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


def get_cfg(args, validate_cfg=validate_cfg_train):
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

def get_stopper(cfg):
    stopper = None
    if cfg.schedule.stopper is None:
        return DummyStopper(cfg)
    if cfg.schedule.stopper.name == 'early_stopper':
        stopper = EarlyStopper(cfg)
    return stopper

def build_scheduler(cfg, optimizer):
    if cfg.schedule.scheduler.name == 'reduce_lr_on_plateau':
        return ReduceLROnPlateauScheduler(cfg, optimizer)
    return DummyScheduler(cfg, optimizer)

def build_optimizer(cfg, model):
    if cfg.model.optimizer.name == 'sgd':
        return SGD(
            model.parameters(),
            lr=cfg.model.optimizer.lr,
            momentum=cfg.model.optimizer.momentum
        )
    if cfg.model.optimizer.name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.model.optimizer.lr,
        )
    raise NotImplementedError()



def _distance_transform(target):
    if torch.all(target == 1):
        padded_target = torch.zeros(tuple((torch.tensor(target.shape) + 2).tolist()))
        padded_target[1:-1, 1:-1] = target
        return scipy.ndimage.distance_transform_edt((padded_target).cpu())[1:-1, 1:-1]
    return scipy.ndimage.distance_transform_edt((target).cpu())


def get_point_prompts(targets):
    point_prompts = []
    point_prompts_labels = []

    for target in targets:
        point_class = 1
        # we assume there's only one target
        assert target.shape[0] == 1
        if torch.all(target[0] == 0):
            point_class = 0
        distance_transform = _distance_transform(target[0] == point_class)
        point_prompt = np.argmax(distance_transform)
        point_prompt = np.unravel_index(point_prompt, distance_transform.shape)
        point_prompts.append([point_prompt])
        point_prompts_labels.append([point_class])
    return torch.tensor(point_prompts).to(targets.device), torch.tensor(point_prompts_labels).to(targets.device)


def get_random_point_prompts(targets) :
    point_prompts = []
    point_prompts_labels = []
    for target in targets:
        point_class = 1
        # we assume there's only one target
        assert target.shape[0] == 1
        if torch.all(target[0] == 0):
            point_class = 0
        xs, ys = np.where((target[0] == point_class).cpu())
        index = np.random.choice(np.arange(len(xs)))
        point_prompt = xs[index], ys[index]
        point_prompts.append([point_prompt])
        point_prompts_labels.append([point_class])
    return torch.tensor(point_prompts).to(targets.device), torch.tensor(point_prompts_labels).to(targets.device)


def train_epoch(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, dataloaders, epoch, logger: EpochLogger, stopper: Stopper):
    train_loader = dataloaders['train']
    model.train()
    total_epoch_train_loss = 0
    for i, batch in tqdm(enumerate(train_loader)):
        samples, targets, classes = batch
        point_prompts, point_prompts_labels = get_random_point_prompts(targets)
        outputs = model(samples, point_prompts, point_prompts_labels)
        loss = call_loss(loss_function, outputs, targets, cfg)
        metrics = call_metrics(metric_functions, outputs, targets, model)
        assert metrics.get('loss') is None
        metrics['loss'] = loss.tolist()
        logger.log_batch_metrics(metrics)
        total_epoch_train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        model.clip_gradients()
        optimizer.step()
    avg_epoch_metrics = logger.get_avg_epoch_metrics()
    logger.log_epoch(epoch)
    stopper.record_metrics(avg_epoch_metrics, 'train')
    return avg_epoch_metrics


def train_iteration(cfg, model: SamWrapper, loss_function: Callable, metric_functions: dict[str, Callable], optimizer,
                    dataloaders, iteration, logger: IterationLogger, stopper: Stopper):
    infinite_train_loader = dataloaders['infinite_train']
    model.train()
    batch = next(infinite_train_loader)
    samples, targets, classes = batch
    point_prompts, point_prompts_labels = get_random_point_prompts(targets)
    outputs = model(samples, point_prompts, point_prompts_labels)
    # print(outputs)
    loss = call_loss(loss_function, outputs, targets, cfg)
    metrics = call_metrics(metric_functions, outputs, targets, model)
    assert metrics.get('loss') is None
    metrics['loss'] = loss.tolist()
    logger.log_iteration_metrics(metrics, iteration)
    stopper.record_metrics(metrics, 'train')
    optimizer.zero_grad()
    loss.backward()
    model.clip_gradients()
    optimizer.step()
    return metrics


def validate_epoch(cfg, model: SamWrapper, loss_function, metric_functions, dataloaders, logger: EpochLogger, stopper: Stopper):
    logger.log('Validation')
    val_loader = dataloaders['val']
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            samples, targets, classes = batch
            point_prompts, point_prompts_labels = get_point_prompts(targets)
            outputs = model(samples, point_prompts, point_prompts_labels)
            loss = call_loss(loss_function, outputs, targets, cfg)
            metrics = call_metrics(metric_functions, outputs, targets, model)
            assert metrics.get('loss') is None
            metrics['loss'] = loss.tolist()
            logger.log_batch_metrics(metrics)
            stopper.record_metrics(metrics, 'val')
    avg_epoch_metrics = logger.get_avg_epoch_metrics()
    logger.log_epoch(1, split='val')
    return avg_epoch_metrics


def train_epochs(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, scheduler, dataloaders, logger, stopper):
    for epoch in range(1, cfg.schedule.epochs + 1):
        train_metrics = train_epoch(cfg, model, loss_function, metric_functions, optimizer, dataloaders, epoch, logger, stopper)
        scheduler.observe_metrics(train_metrics, 'train')
        if epoch % cfg.schedule.val_interval == 0:
            validation_metrics = validate_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger, stopper)
            scheduler.observe_metrics(validation_metrics, 'val')
        if stopper.should_stop():
            break

def train_iterations(cfg, model: SamWrapper, loss_function, metric_functions, optimizer, scheduler, dataloaders, logger, stopper):
    dataloaders['infinite_train'] = InfiniteIterator(dataloaders['train'])
    for iteration in range(1, cfg.schedule.iterations + 1):
        train_metrics = train_iteration(cfg, model, loss_function, metric_functions, optimizer, dataloaders, iteration, logger, stopper)
        scheduler.observe_metrics(train_metrics, 'train')
        if iteration % cfg.schedule.val_interval == 0:
            validation_metrics = validate_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger, stopper)
            scheduler.observe_metrics(validation_metrics, 'val')
        if stopper.should_stop():
            break

def seed(cfg):
    _seed = cfg.seed
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

def log_parameter_counts(model: SamWrapper, logger):
    model.train()
    total_params = get_param_count(model.model)
    trainable_params = get_param_count_trainable_recursively(model.model)
    logger.log(f'Parameters: total: {total_params}, trainable: {trainable_params}')


def train(cfg):
    logger = get_logger(cfg)
    store_cfg(cfg, logger)
    seed(cfg)
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg, logger)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    stopper = get_stopper(cfg)
    loss_function = build_loss_function(cfg)
    metric_functions = build_metric_functions(cfg)
    logger.log('Training')
    log_parameter_counts(model, logger)
    if cfg.schedule.iterations is not None:
        train_iterations(cfg, model, loss_function, metric_functions, optimizer, scheduler, dataloaders, logger, stopper)
    elif cfg.schedule.epochs is not None:
        train_epochs(cfg, model, loss_function, metric_functions, optimizer, scheduler, dataloaders, logger, stopper)
    checkpoint(cfg, model, optimizer)


def main():
    # parse config
    args = parse_args()
    cfg = get_cfg(args)
    with torch.autograd.detect_anomaly(check_nan=True):
        train(cfg)


if __name__ == '__main__':
    main()
