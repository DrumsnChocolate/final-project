import os.path as osp
from typing import Optional

import torch
from prodict import Prodict

from finetune.models.sam_wrapper import SamWrapper

MODEL_STATE_DICT = 'model'
OPTIMIZER_STATE_DICT = 'optimizer'



def create_checkpoint(checkpoint_path: str, model: SamWrapper, optimizer):
    torch.save({
        MODEL_STATE_DICT: model.state_dict(),
        OPTIMIZER_STATE_DICT: optimizer.state_dict(),
    }, checkpoint_path)


def load_checkpoint(checkpoint_path: str, model: SamWrapper, optimizer: Optional[torch.optim.Optimizer] = None) -> tuple[SamWrapper, Optional[torch.optim.Optimizer]]:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT])
    return model, optimizer


def checkpoint(cfg: Prodict, model: SamWrapper, optimizer):
    checkpoint_path = osp.join(cfg.out_dir, cfg.timestamp, 'checkpoint.pth')
    create_checkpoint(checkpoint_path, model, optimizer)


def load(cfg: Prodict, model: SamWrapper, optimizer: Optional[torch.optim.Optimizer] = None):
    checkpoint_path = osp.join(cfg.out_dir, cfg.timestamp, 'checkpoint.pth')
    return load_checkpoint(checkpoint_path, model, optimizer)
