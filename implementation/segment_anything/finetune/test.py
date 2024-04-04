import argparse

from finetune.checkpoint import load_checkpoint, load
from finetune.configs.config_options import DictAction
from finetune.loss import build_loss_function
from models import build_model
from datasets.loaders import build_dataloaders
from train import parse_args, get_cfg, get_logger, test_epoch, store_cfg
from metrics import build_metric_functions
import os.path as osp

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
    model = build_model(cfg, logger)
    model, _ = load(cfg, model)  # pass no optimizer, so we don't need the second return value
    dataloaders = build_dataloaders(cfg)
    loss_function = build_loss_function(cfg)
    metric_functions = build_metric_functions(cfg)
    test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def main():
    args = parse_args()
    args.config = osp.join(args.train_dir, 'config.yaml')
    cfg = get_cfg(args)
    cfg.timestamp = args.train_dir.split('/')[-1]
    cfg.sub_dir = 'test'
    test(cfg)

if __name__ == '__main__':
    main()
