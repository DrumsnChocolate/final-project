import os
import random

import numpy as np
import torch

from implementation.visual_prompt_tuning.launch import default_argument_parser
from tune_vtab import get_lrwd_range

from src.configs.config import get_cfg
from src.utils.file_io import PathManager

DATA2CLS = {
    "cbis": 2,
}


def seed(cfg):
    SEED = cfg.SEED
    if SEED is None:
        # default seed
        SEED = 218
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # TODO: complete this with more seeding calls to make sure everything becomes reproducible


def construct_output_dir(cfg):
    output_folder = os.path.join(
        cfg.OUTPUT_DIR,
        cfg.DATA.NAME,
        cfg.DATA.FEATURE,
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


def setup(args, lr, wd, seed=None):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SEED = seed
    cfg.RUN_N_TIMES = 1
    cfg.MODEL.SAVE_CKPT = False
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = wd
    cfg.OUTPUT_DIR = construct_output_dir(cfg)
    return cfg

def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # main training / eval actions here
    seed(cfg)


def explore_lrwd_range(args):
    lr_range, wd_range = get_lrwd_range(args)

    for lr in sorted(lr_range, reverse=True):
        for wd in sorted(wd_range, reverse=True):
            try:
                cfg = setup(args, lr, wd)
            except ValueError:
                # already ran
                continue
            train(cfg, args)


def get_best_lrwd(args):
    raise NotImplementedError()


def main(args):
    explore_lrwd_range(args)
    lr, wd = get_best_lrwd(args)
    random_seeds = np.random.randn(5)
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