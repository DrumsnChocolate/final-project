#!/usr/bin/env python3
"""
major actions here for training VTAB datasets: use val200 to find best lr/wd, and retrain on train800val200, report results on test
"""
import glob
import numpy as np
import os
import torch
import warnings
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager


from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")
DATA2CLS = {
    'caltech101': 102,
    'cifar(num_classes=100)': 100,
    'dtd': 47,
    'oxford_flowers102': 102,
    'oxford_iiit_pet': 37,
    'patch_camelyon': 2,
    'sun397': 397,
    'svhn': 10,
    'resisc45': 45,
    'eurosat': 10,
    'dmlab': 6,
    'kitti(task="closest_vehicle_distance")': 4,
    'smallnorb(predicted_attribute="label_azimuth")': 18,
    'smallnorb(predicted_attribute="label_elevation")': 9,
    'dsprites(predicted_attribute="label_x_position",num_classes=16)': 16,
    'dsprites(predicted_attribute="label_orientation",num_classes=16)': 16,
    'clevr(task="closest_object_distance")': 6,
    'clevr(task="count_all")': 8,
    'diabetic_retinopathy(config="btgraham-300")': 5
}

DEFAULT_SEED = 0

# todo: make below functions into a class, so we can more easily override things

def find_best_lrwd(files, data_name):
    t_name = "val_" + data_name
    best_lr = None
    best_wd = None
    best_val_acc = -1
    for f in files:
        try:
            results_dict = torch.load(f, "cpu")
            epoch = len(results_dict) - 1
            val_result = results_dict[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            val_result = float(val_result)
        except Exception as e:
            print(f"Encounter issue: {e} for file {f}")
            continue

        if val_result == best_val_acc:
            frag_txt = f.split("/run")[0]
            cur_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            cur_wd = float(frag_txt.split("_wd")[-1])
            if best_lr is not None and cur_lr < best_lr:
                # get the smallest lr to break tie for stability
                best_lr = cur_lr
                best_wd = cur_wd
                best_val_acc = val_result

        elif val_result > best_val_acc:
            best_val_acc = val_result
            frag_txt = f.split("/run")[0]
            best_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            best_wd = float(frag_txt.split("_wd")[-1])
    return best_lr, best_wd


def get_loaders(cfg, logger, test=True):
    # support two training paradims:
    # 1) train / val / test, using val to tune
    # 2) train / val: for imagenet
    logger.info("Loading training data...")
    if test:
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
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

def seed(cfg):
    SEED = cfg.SEED
    assert SEED is not None, "No seed was configured"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # todo: also make the dataset loader deterministic using a generator? Is this necessary?


def construct_output_dir(cfg, test=True):
    transfer_method = cfg.MODEL.TRANSFER_TYPE
    if transfer_method == "prompt":
        transfer_method = f"prompt{cfg.MODEL.PROMPT.NUM_TOKENS}"
    output_folder = os.path.join(
        cfg.OUTPUT_DIR,
        cfg.DATA.NAME,
        cfg.DATA.FEATURE,
        transfer_method,
        f"crop{cfg.DATA.CROPSIZE}",
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


def setup_part(args, lr, wd, seed=DEFAULT_SEED):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SEED = seed
    cfg.RUN_N_TIMES = 1
    cfg.MODEL.SAVE_CKPT = False
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = wd
    return cfg

def setup(args, lr, wd, test=True, seed=DEFAULT_SEED):
    cfg = setup_part(args, lr, wd, seed=seed)
    cfg.OUTPUT_DIR = construct_output_dir(cfg, test=test)
    return cfg


def get_lrwd_range(args):
    print(args)
    lr_range = None
    wd_range = None

    if args.train_type == "finetune":
        lr_range = [0.001, 0.0001, 0.0005, 0.005]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "finetune_resnet":
        lr_range = [
            0.0005, 0.00025,
            0.5, 0.25, 0.05, 0.025, 0.005, 0.0025,
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "linear":
        lr_range = [
            50.0, 25., 10.0,
            5.0, 2.5, 1.0,
            0.5, 0.25, 0.1, 0.05
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "linear_mae":
        lr_range = [
            50.0, 25., 10.0,
            5.0, 2.5, 1.0,
            0.5, 0.25, 0.1, 0.05,
            0.025, 0.005, 0.0025,
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "prompt":
        lr_range = [
            5.0, 2.5, 1.0,
            50.0, 25., 10.0,
            0.5, 0.25, 0.1, 0.05
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "prompt_largerlr":
        lr_range = [
            500, 1000, 250., 100.0,
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "prompt_resnet":
        lr_range = [
            0.05, 0.025, 0.01, 0.5, 0.25, 0.1,
            1.0, 2.5, 5.
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    return lr_range, wd_range


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


def get_best_lrwd(args):
    cfg = setup_part(args, None, None)
    transfer_method = cfg.MODEL.TRANSFER_TYPE
    if transfer_method == "prompt":
        transfer_method = f"prompt{cfg.MODEL.PROMPT.NUM_TOKENS}"
    # this might not be cross-platform-friendly, not sure if glob works on windows
    files = glob.glob(f"{cfg.OUTPUT_DIR}/"
                      f"{cfg.DATA.NAME}/"
                      f"{cfg.DATA.FEATURE}/"
                      f"{transfer_method}/"
                      f"crop{cfg.DATA.CROPSIZE}/"
                      "val/"
                      "*/"  # seed
                      "*/"  # hyperparams
                      "run1/"
                      "eval_results.pth")
    lr, wd = find_best_lrwd(files, cfg.DATA.NAME)
    return lr, wd


def main(args):
    explore_lrwd_range(args)
    lr, wd = get_best_lrwd(args)
    random_seeds = np.random.randint(10000, size=5)
    for seed in random_seeds:
        seed = int(seed)  # convert from np.int64 to int
        try:
            cfg = setup(args, lr, wd, seed=seed)
        except ValueError:
            # already ran
            continue
        train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
