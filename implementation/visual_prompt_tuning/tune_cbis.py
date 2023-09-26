import glob
import os
import random

import numpy as np
import torch


import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model

from launch import default_argument_parser, logging_train_setup
from tune_vtab import explore_lrwd_range, get_best_lrwd, seed, setup, get_lrwd_range

# todo: actually start using this constant
DATA2CLS = {
    "cbis": 2,
}


def train(cfg, args, test=True):
    cfg.freeze()
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # main training / eval actions here
    seed(cfg)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger, test=test)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)
    logger.info("Setting up Evaluator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)
    trainer.train_classifier(train_loader, val_loader, test_loader)
    # save the evaluation results
    torch.save(
        evaluator.results,
        os.path.join(cfg.OUTPUT_DIR, "eval_results.pth")
    )


def get_loaders(cfg, logger, test=True):
    # todo: do we want to use the concept of trainval? like done with vtab?
    #  (because of the small size of cbis dataset, this could be useful)
    logger.info("Loading training data...")
    train_loader = data_loader.construct_train_loader(cfg)
    logger.info("Loading validation data...")
    val_loader = data_loader.construct_val_loader(cfg)
    test_loader = None
    if test:
        logger.info("Loading test data...")
        test_loader = data_loader.construct_test_loader(cfg)
    assert train_loader is not None, "No train loader available. EXIT"
    assert val_loader is not None, "No validation loader available. EXIT"
    assert not test or test_loader is not None, "No test loader available. EXIT"
    return train_loader, val_loader, test_loader


def explore_lrwd_range(args):
    lr_range, wd_range = get_lrwd_range(args)

    for lr in sorted(lr_range, reverse=True):
        for wd in sorted(wd_range, reverse=True):
            try:
                cfg = setup(args, lr, wd, test=False)
            except ValueError:
                # already ran
                continue
            train(cfg, args, test=False)


def main(args):
    explore_lrwd_range(args)
    lr, wd = get_best_lrwd(args)
    random_seeds = np.random.randint(10000, size=5)
    for seed in random_seeds:
        try:
            cfg = setup(args, lr, wd, seed=seed)
        except ValueError:
            # already ran
            continue
        train(cfg, args)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)