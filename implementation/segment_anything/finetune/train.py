import argparse
from typing import Callable

from prodict import Prodict
from torch.optim import SGD
from yaml import load, Loader
from configs.config_options import DictAction
from configs.config_validation import validate_cfg
from finetune.loss import build_loss_function, call_loss
from logger import Logger
from metrics import call_metrics
from models import build_model, call_model
from datasets.loaders import build_dataloaders
from segment_anything.modeling import Sam

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
    return cfg

def cfg_to_prodict(cfg):
    cfg = Prodict.from_dict(cfg)
    cfg.data.preprocess = [Prodict.from_dict(step) for step in cfg.data.preprocess]
    if type(cfg.model.loss) == list:
        cfg.model.loss = [Prodict.from_dict(loss_item) for loss_item in cfg.model.loss]
    cfg.model.metrics = [Prodict.from_dict(metric) for metric in cfg.model.metrics]
    return cfg

def get_cfg(args):
    cfg = cfg_to_prodict(get_cfg_dict(args))
    validate_cfg(cfg)
    return cfg


def get_logger(cfg):
    logger = Logger(log=print)
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


def train_epoch(cfg, model: Sam, loss_function, metric_functions, optimizer, dataloaders, epoch, logger):
    train_loader = dataloaders['train']
    model.train()
    total_epoch_train_loss = 0
    epoch_train_metrics = []
    total_epoch_train_samples = 0
    for i, batch in enumerate(train_loader):
        samples, targets = batch
        outputs = call_model(model, samples, logger)
        loss = call_loss(loss_function, outputs, targets)
        metrics = call_metrics(metric_functions, outputs, targets)
        # todo: log metrics and store them somewhere, including loss
        logger.log(f'Epoch {epoch}, batch {i}, train loss {loss / len(samples)}')
        total_epoch_train_loss += loss
        epoch_train_metrics.append(metrics)
        total_epoch_train_samples += len(samples)
        loss.backward()
        optimizer.step()
    # todo: log metrics and store them somewhere, including loss
    logger.log(f'Epoch {epoch}, average train loss {total_epoch_train_loss / total_epoch_train_samples}')


def train_iteration(cfg, model: Sam, loss_function: Callable, metric_functions: dict[str, Callable], optimizer, dataloaders, iteration, logger):
    infinite_train_loader = dataloaders['infinite_train']
    model.train()
    batch = next(infinite_train_loader)
    samples, targets = batch
    outputs = call_model(model, samples, logger)
    loss = call_loss(loss_function, outputs, targets)
    metrics = call_metrics(metric_functions, outputs, targets)
    # todo: log metrics and store them somewhere, including loss
    logger.log(f'Iteration {iteration}, train loss {loss / len(samples)}')
    loss.backward()
    optimizer.step()


def validate_epoch(cfg, model: Sam, loss_function, metric_functions, dataloaders, logger):
    val_loader = dataloaders['val']
    model.eval()
    total_val_loss = 0
    val_metrics = []
    total_val_samples = 0
    for i, batch in enumerate(val_loader):
        samples, targets = batch
        outputs = call_model(model, samples, logger)
        loss = call_loss(loss_function, outputs, targets)
        metrics = call_metrics(metric_functions, outputs, targets)
        total_val_loss += loss
        val_metrics.append(metrics)
        total_val_samples += len(samples)
    # todo: log metrics and store them somewhere, including loss
    logger.log(f'Validation, average val loss {total_val_loss / total_val_samples}')


def test_epoch(cfg, model: Sam, loss_function, metric_functions, dataloaders, logger):
    test_loader = dataloaders['test']
    model.eval()
    total_test_loss = 0
    test_metrics = []
    total_test_samples = 0
    for i, batch in enumerate(test_loader):
        samples, targets = batch
        outputs = call_model(model, samples, logger)
        loss = call_loss(loss_function, outputs, targets)
        metrics = call_metrics(metric_functions, outputs, targets)
        total_test_loss += loss
        test_metrics.append(metrics)
        total_test_samples += len(samples)
        raise NotImplementedError()
    # todo: log metrics and store them somewhere, including loss
    logger.log(f'Test, average test loss {total_test_loss/total_test_samples}')


def train_epochs(cfg, model: Sam, loss_function, metric_functions, optimizer, dataloaders, logger):
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


def train_iterations(cfg, model: Sam, loss_function, metric_functions, optimizer, dataloaders, logger):
    dataloaders['infinite_train'] = InfiniteIterator(dataloaders['train'])
    for iteration in range(cfg.schedule.iterations):
        train_iteration(cfg, model, loss_function, metric_functions, optimizer, dataloaders, iteration, logger)
        if (iteration+1) % cfg.schedule.val_interval != 0:
            continue
        validate_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)
    test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def train(cfg):
    logger = get_logger(cfg) # todo: construct something that can .log() to a file? and possibly make it efficient? idk man
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    # todo: optionally include background in loss computation (not for ade20k, yes for cbis)
    loss_function = build_loss_function(cfg)  # todo: construct (20*focal + 1*dice)/21 as a sum over batch?
    metric_functions = {}  # todo: construct iou as a sum over batch?
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
