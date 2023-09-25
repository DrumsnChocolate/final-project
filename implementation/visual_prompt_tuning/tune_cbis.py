import glob
import os
import random

import numpy as np
import torch

from implementation.visual_prompt_tuning.launch import default_argument_parser, logging_train_setup
from tune_vtab import get_lrwd_range, find_best_lrwd

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

DATA2CLS = {
    "cbis": 2,
}

DEFAULT_SEED = 0


def seed(cfg):
    SEED = cfg.SEED
    if SEED is None:
        SEED = DEFAULT_SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # TODO: complete this with more seeding calls to make sure everything becomes reproducible

def get_best_lrwd(args):
    cfg = setup_part(args, None, None, seed=None)
    transfer_method = cfg.DATA.TRANSFER_TYPE
    if transfer_method == "prompt":
        transfer_method = f"prompt{cfg.PROMPT.NUM_TOKENS}"
    # this might not be cross-platform-friendly, not sure if glob works on windows
    files = glob.glob(f"{cfg.OUTPUT_DIR}/"
                      f"{cfg.DATA.NAME}/"
                      f"{cfg.DATA.FEATURE}/"
                      f"{transfer_method}/"
                      f"{cfg.DATA.CROPSIZE}/"
                      "val/"
                      "*/"  # seed
                      "*/"  # hyperparams
                      "run1/"
                      "eval_results.pth")
    lr, wd = find_best_lrwd(files, cfg.DATA.NAME)
    return lr, wd

def construct_output_dir(cfg, test=True):
    transfer_method = cfg.DATA.TRANSFER_TYPE
    if transfer_method == "prompt":
        transfer_method = f"prompt{cfg.PROMPT.NUM_TOKENS}"
    output_folder = os.path.join(
        cfg.OUTPUT_DIR,
        cfg.DATA.NAME,
        cfg.DATA.FEATURE,
        transfer_method,
        cfg.DATA.CROPSIZE,
        "test" if test else "val",
        f"seed{cfg.SEED}",
        f"lr{cfg.SOLVER.BASE_LR}_wd{cfg.SOLVER.WEIGHT_DECAY}",
    )

    for run in range(1, cfg.RUN_N_TIMES+1):
        output_path = os.path.join(output_folder, f"run{run}")
        if PathManager.exists(output_path):
            continue
        PathManager.mkdirs(output_path)
        return output_path
    # at the end of the loop without returning, so failed
    raise ValueError(f"Alread run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

def setup_part(args, lr, wd, seed=None):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SEED = seed
    cfg.RUN_N_TIMES = 1
    cfg.MODEL.SAVE_CKPT = False
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = wd
    return cfg

def setup(args, lr, wd, test=True, seed=None):
    cfg = setup_part(args, lr, wd, seed=seed)
    cfg.OUTPUT_DIR = construct_output_dir(cfg, test=test)
    return cfg


def get_loaders(cfg, logger, test=True):
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


def train(cfg, args, test=True):
    cfg.freeze()
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # main training / eval actions here
    seed(cfg)

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
    for seed in enumerate(random_seeds):
        try:
            cfg = setup(args, lr, wd, seed=seed)
        except ValueError:
            # already ran
            continue
        train(cfg, args)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)