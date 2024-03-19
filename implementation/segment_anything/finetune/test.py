import argparse

from finetune.configs.config_options import DictAction
from finetune.loss import build_loss_function
from models import build_model
from datasets.loaders import build_dataloaders
from train import parse_args, get_cfg, get_logger, test_epoch, store_cfg
from metrics import build_metric_functions

def parse_args():
    parser = argparse.ArgumentParser(description='Test a segment anything model')
    parser.add_argument('train_dir', help='directory containing the trained model')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
    )
    return parser.parse_args()

def test(cfg):
    logger = get_logger(cfg)
    store_cfg(cfg, logger)
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg, logger)
    loss_function = build_loss_function(cfg)
    metric_functions = build_metric_functions(cfg)
    test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def main():
    args = parse_args()
    cfg = get_cfg(args)
    test(cfg)

if __name__ == '__main__':
    main()
