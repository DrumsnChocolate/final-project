import argparse

from prodict import Prodict
from torch.optim import SGD
from yaml import load, Loader
from configs.config_options import DictAction
from configs.config_validation import validate_cfg
from logger import Logger
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


def get_cfg(args):
    cfg = load_cfg(args.config)
    bases = cfg.get('_bases_', None)
    if bases:
        for base in bases:
            cfg = override_cfg(load_cfg(base), cfg)
    if args.cfg_options:
        cfg = override_cfg(cfg, args.cfg_options)
    return cfg

def cfg_to_prodict(cfg):
    cfg = Prodict.from_dict(cfg)
    cfg.data.preprocess = [Prodict.from_dict(step) for step in cfg.data.preprocess]
    return cfg




def build_optimizer(cfg, model):
    if cfg.model.optimizer.name == 'sgd':
        return SGD(
            model.parameters(),
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.wd,
            momentum=cfg.model.optimizer.momentum
        )
    raise NotImplementedError()

def train_epoch(cfg, model: Sam, loss_function, eval_function, optimizer, dataloaders, epoch, logger):
    train_loader = dataloaders['train']
    model.train()
    total_epoch_train_loss = 0
    total_epoch_train_eval = 0
    total_epoch_train_samples = 0
    for i, batch in enumerate(train_loader):
        samples, targets = batch
        outputs = call_model(model, samples, logger)
        loss = loss_function(outputs, targets)
        evaluation = eval_function(outputs, targets)
        logger.log(f'Epoch {epoch}, batch {i}, train loss {loss / len(samples)}, train evaluation {evaluation / len(samples)}')
        total_epoch_train_loss += loss
        total_epoch_train_eval += evaluation
        total_epoch_train_samples += len(samples)
        loss.backward()
        optimizer.step()
    logger.log(f'Epoch {epoch}, average train loss {total_epoch_train_loss / total_epoch_train_samples}')

def train_iteration(cfg, model: Sam, loss_function, eval_function, optimizer, dataloaders, iteration, logger):
    infinite_train_loader = dataloaders['infinite_train']
    model.train()
    batch = next(infinite_train_loader)
    samples, targets = batch
    outputs = call_model(model, samples, logger)
    loss = loss_function(outputs, targets)
    evaluation = eval_function(outputs, targets)
    logger.log(f'Iteration {iteration}, train loss {loss / len(samples)}, train evaluation {evaluation / len(samples)}')
    loss.backward()
    optimizer.step()

def validate_epoch(cfg, model: Sam, loss_function, eval_function, dataloaders, epoch, logger):
    val_loader = dataloaders['val']
    model.eval()
    total_val_loss = 0
    total_val_eval = 0
    total_val_samples = 0
    for i, batch in enumerate(val_loader):
        samples, targets = batch
        outputs = call_model(model, samples, logger)
        loss = loss_function(outputs, targets)
        evaluation = eval_function(outputs, targets)
        total_val_loss += loss
        total_val_eval += evaluation
        total_val_samples += len(samples)
    logger.log(f'Epoch {epoch}, average val loss {total_val_loss / total_val_samples}, average val evaluation {total_val_eval / total_val_samples}')


def validate_iteration(cfg, model: Sam, loss_function, eval_function, dataloaders, iteration, logger):
    val_loader = dataloaders['val']
    model.eval()
    total_val_loss = 0
    total_val_eval = 0
    total_val_samples = 0
    for i, batch in enumerate(val_loader):
        samples, targets = batch
        outputs = call_model(model, samples, logger)
        loss = loss_function(outputs, targets)
        evaluation = eval_function(outputs, targets)
        total_val_loss += loss
        total_val_eval += evaluation
        total_val_samples += len(samples)
    logger.log(f'Iteration {iteration}, average val loss {total_val_loss / total_val_samples}, average val evaluation {total_val_eval / total_val_samples}')


def test(cfg, model: Sam, loss_function, eval_function, dataloaders, logger):
    test_loader = dataloaders['test']
    model.eval()
    total_test_loss = 0
    total_test_eval = 0
    total_test_samples = 0
    for i, batch in enumerate(test_loader):
        samples, targets = batch
        outputs = call_model(model, samples, logger)
        loss = loss_function(outputs, targets)
        evaluation = eval_function(outputs, targets)
        total_test_loss += loss
        total_test_eval += evaluation
        total_test_samples += len(samples)
    logger.log(f'Average test loss {total_test_loss/total_test_samples}')


def train_epochs(cfg, model: Sam, loss_function, eval_function, optimizer, dataloaders, logger):
    for epoch in range(cfg.schedule.epochs):
        train_epoch(cfg, model, loss_function, eval_function, optimizer, dataloaders, epoch, logger)
        validate_epoch(cfg, model, loss_function, eval_function, dataloaders, epoch, logger)
    test(cfg, model, loss_function, eval_function, dataloaders, logger)


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


def train_iterations(cfg, model: Sam, loss_function, eval_function, optimizer, dataloaders, logger):
    dataloaders['infinite_train'] = InfiniteIterator(dataloaders['train'])
    for iteration in range(cfg.schedule.iterations):
        train_iteration(cfg, model, loss_function, eval_function, optimizer, dataloaders, iteration, logger)
        if (iteration+1) % cfg.schedule.val_interval == 0:
            validate_iteration(cfg, model, loss_function, eval_function, dataloaders, iteration, logger)
    test(cfg, model, loss_function, eval_function, dataloaders, logger)


def train(cfg):
    logger = Logger(log=print)  # todo: construct something that can .log() to a file? and possibly make it efficient? idk man
    logger.log(cfg)
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    loss_function = None  # todo: construct crossentropy as a sum over batch?
    eval_function = None  # todo: construct iou as a sum over batch?
    if cfg.schedule.iterations is not None:
        train_iterations(cfg, model, loss_function, eval_function, optimizer, dataloaders, logger)
    elif cfg.schedule.epochs is not None:
        train_epochs(cfg, model, loss_function, eval_function, optimizer, dataloaders, logger)




def main():
    # parse config
    args = parse_args()
    cfg = cfg_to_prodict(get_cfg(args))
    validate_cfg(cfg)
    train(cfg)


if __name__ == '__main__':
    main()