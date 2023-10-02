import os
from collections import Counter

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.io import read_image, ImageReadMode

from ..transforms import get_transforms
from ...utils import logging
logger = logging.get_logger("visual_prompt")

DATASETS = [
    "cbis-ddsm",
    "vindr",
]

class CBISDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' not supported for {cfg.DATA.NAME} dataset"
        logger.info(f"Constructing {cfg.DATA.NAME} dataset {split}...")

        self.cfg = cfg
        self._split = split
        # todo: figure out what this is for
        self.name = cfg.DATA.NAME
        # todo: figure out how to use this dir, exactly
        self.data_dir = cfg.DATA.DATAPATH
        # todo: figure out if this percentage is ever used anywhere, and if we want to use it
        self.data_percentage = cfg.DATA.PERCENTAGE
        # todo: not sure about the transforms, we'll see
        self.transform = get_transforms(split, cfg.DATA.CROPSIZE)
        self.image_labels = self.get_image_labels()
        self._class_to_id = self.get_class_to_id()


    def __getitem__(self, index):
        # only 8 bit is straightforward, 16bit provides more detail but is harder to use; how to convert to 3xHxW?
        image_path = os.path.join(self.data_dir, "cbis-ddsm", "multiinstance_data_8bit", self.image_labels.loc[index]["ShortPath"])
        # image = read_image(image_path, mode=ImageReadMode.GRAY)
        image = Image.open(image_path)
        image = self.transform(image)
        label = self._class_to_id[self.image_labels.loc[index]['ImageLabel']]
        sample = {
            "image": image,
            "label": label,
        }
        return sample

    def __len__(self):
        return len(self.image_labels)

    def get_image_labels(self):
        all_labels = pd.read_csv(os.path.join(self.data_dir, "cbis-ddsm", "cbis-ddsm_singleinstance_groundtruth.csv"), sep=';')
        if self._split == "test":
            test = all_labels[all_labels["ShortPath"].str.contains("Test")]
            test = test.reset_index(drop=True)
            return test
        original_train = all_labels[all_labels["ShortPath"].str.contains("Train")]
        train, val = train_test_split(original_train, test_size=0.1, shuffle=True, random_state=218, stratify=original_train["ImageLabel"])
        if self._split == "train":
            train = train.reset_index(drop=True)
            return pd.DataFrame(train)
        elif self._split == "val":
            val = val.reset_index(drop=True)
            return pd.DataFrame(val)

    def get_class_to_id(self):
        return {label: i for i, label in enumerate(self.get_classes())}

    def get_info(self):
        num_imgs = len(self.image_labels)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return 2

    def get_classes(self):
        return sorted(list(set(self.image_labels["ImageLabel"])))

    def get_class_ids(self):
        return [self._class_to_id[c] for c in self.get_classes()]

    def get_class_weights(self, weight_type):
        if self._split != "train":
            raise ValueError(f"only getting training class distribution, got split {self._split} instead")
        if weight_type == "none":
            return [1.0] * self.get_class_num()
        id2counts = Counter(self.image_labels["ImageLabel"])
        assert len(id2counts) == self.get_class_num()
        num_per_cls = np.array([id2counts[i] for i in self.get_class_ids()])
        mu = 0
        if weight_type == 'inv':
            mu = -1.0
        if weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(weight_list, np.linalg.norm(weight_list, 1)) * self.get_class_num()
        return weight_list.tolist()






