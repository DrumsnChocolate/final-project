import argparse

from finetune.configs.config_options import DictAction
from finetune.configs.config_validation import validate_cfg_test
from finetune.datasets.loaders import build_dataloaders
from finetune.loss import build_loss_function
from finetune.metrics import build_metric_functions
from finetune.models.build_model import build_model
from finetune.test import test_epoch, get_test_logger
from finetune.train import get_cfg, store_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Apply a segment anything model zero-shot')
    parser.add_argument('config', help='config file containing the test instructions')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
    )
    return parser.parse_args()

def zero_shot(cfg):
    logger = get_test_logger(cfg)
    store_cfg(cfg, logger)
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg, logger)
    loss_function = build_loss_function(cfg)
    metric_functions = build_metric_functions(cfg)
    test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def main():
    args = parse_args()
    cfg = get_cfg(args, validate_cfg=validate_cfg_test)
    cfg.sub_dir = 'test'
    zero_shot(cfg)

if __name__ == '__main__':
    main()
