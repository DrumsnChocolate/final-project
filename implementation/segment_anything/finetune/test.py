import argparse
import os.path as osp

import torch
from tqdm import tqdm

from finetune.checkpoint import load
from finetune.configs.config_options import DictAction
from finetune.configs.config_validation import validate_cfg_test
from finetune.datasets.loaders import build_dataloaders
from finetune.logger import EpochLogger
from finetune.loss import build_loss_function, call_loss
from finetune.metrics import build_metric_functions, call_metrics
from finetune.models.build_model import build_model
from finetune.models.sam_wrapper import SamWrapper
from finetune.train import get_cfg, store_cfg, get_point_prompts, dump_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Test a segment anything model')
    parser.add_argument('train_dir', help='directory containing the trained model')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
    )
    parser.add_argument('--sub-dir', default='test', help='subdirectory of the train_dir to store the test results')
    return parser.parse_args()


def get_test_logger(cfg):
    logger = EpochLogger(cfg)
    logger.log(dump_cfg(cfg), to_file=False)
    return logger


def test_epoch(cfg, model: SamWrapper, loss_function, metric_functions, dataloaders, logger: EpochLogger):
    logger.log('Testing')
    test_loader = dataloaders['test']
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            samples, targets, classes = batch
            point_prompts, point_prompts_labels = get_point_prompts(targets)
            outputs = model(samples, point_prompts, point_prompts_labels)
            loss = call_loss(loss_function, outputs, targets, cfg)
            metrics = call_metrics(metric_functions, outputs, targets, model)
            assert metrics.get('loss') is None
            metrics['loss'] = loss.tolist()
            logger.log_batch_metrics(metrics)
            total_test_loss += loss
    logger.log_epoch(1, split='test')


def test(cfg):
    logger = get_test_logger(cfg)
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
    cfg = get_cfg(args, validate_cfg=validate_cfg_test)
    cfg.timestamp = args.train_dir.split('/')[-1]
    cfg.sub_dir = args.sub_dir
    test(cfg)


if __name__ == '__main__':
    main()
