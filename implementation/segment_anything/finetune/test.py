import argparse

from prodict import Prodict
from torch.optim import SGD
from yaml import load, Loader
from configs.config_options import DictAction
from configs.config_validation import validate_cfg
from finetune.loss import build_loss_function
from logger import Logger
from models import build_model, call_model
from datasets.loaders import build_dataloaders
from segment_anything.modeling import Sam
from train import parse_args, get_cfg, get_logger, test_epoch


def test(cfg):
    logger = get_logger(cfg)
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg)
    loss_function = build_loss_function(cfg)
    metric_functions = {}
    test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def main():
    args = parse_args()
    cfg = get_cfg(args)
    test(cfg)

if __name__ == '__main__':
    main()
