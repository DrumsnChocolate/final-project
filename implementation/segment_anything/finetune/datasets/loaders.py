import os

import torch
from PIL import Image
from prodict import Prodict


class ADE20KDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: Prodict, split: str):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' not supported for {cfg.data.name} dataset"
        self.cfg = cfg
        self._split = split
        self.image_names = self.get_image_names()
        self.image_extension =

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.cfg.data.root, self.cfg.data[self._split].image_dir, image_name + self.cfg.data.image_extension)
        annotation_path = os.path.join(self.cfg.data.root, self.cfg.data[self._split].annotation_dir, image_name + self.cfg.data.annotation_extension)
        image = Image.open(image_path)
        annotation =


    def get_image_names(self):



def build_dataloaders(cfg):
    # todo: implement
    return {'train': None, 'val': None, 'test': None}
