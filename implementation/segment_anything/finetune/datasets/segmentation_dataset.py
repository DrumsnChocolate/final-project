import os

import torch
from prodict import Prodict
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as F, InterpolationMode
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, cfg: Prodict, split: str):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' not supported for {cfg.data.name} dataset"
        self.cfg = cfg
        self._split = split
        self.image_names = self.get_image_names()

    def get_image_path(self, image_name):
        return os.path.join(self.cfg.data.root, self.cfg.data[self._split].image_dir,
                            image_name + self.cfg.data.image_extension)

    def get_annotation_path(self, image_name):
        return os.path.join(self.cfg.data.root, self.cfg.data[self._split].annotation_dir,
                            image_name + self.cfg.data.annotation_extension)

    def preprocess(self, image):
        preprocess = self.cfg.data.preprocess
        if preprocess is None:
            return image
        for step in preprocess:
            if step.name == 'resize':
                mode = step.mode if step.mode is not None else 'bilinear'
                interpolation = InterpolationMode[mode.upper()]
                image = F.resize(image, size=step.dimensions, interpolation=interpolation)
            else:
                raise NotImplementedError(f'preprocessing step {step.name} has not been implemented')
        return image

    def preprocess_batch(self, images):
        return [self.preprocess(image) for image in images]

    def get_item_by_index(self, index):
        image_name = self.image_names[index]
        image_path = self.get_image_path(image_name)
        annotation_path = self.get_annotation_path(image_name)
        image = read_image(image_path)
        # using ImageReadMode.GRAY to be sure, though this should be done automatically already
        annotation = read_image(annotation_path, mode=ImageReadMode.GRAY)
        image, annotation = self.preprocess(image), self.preprocess(annotation)
        return image, annotation

    def get_item_by_slice(self, _slice):
        """
        Returns a tuple of a batch of images and a batch of annotations.
        The images are stacked into a single tensor, and so are the annotations.
        The dimensions of the returned images are [B, C, H, W].
        The dimensions of the returned annotations are [B, 1, H, W].

        Arguments:
            _slice (slice): slice object that indicates which indices to obtain.
        """
        image_names = self.image_names[_slice]
        image_paths = [self.get_image_path(image_name) for image_name in image_names]
        annotation_paths = [self.get_annotation_path(image_name) for image_name in image_names]
        images = [read_image(image_path) for image_path in image_paths]
        # using ImageReadMode.GRAY to be sure, though this should be done automatically already
        annotations = [read_image(annotation_path, mode=ImageReadMode.GRAY) for annotation_path in annotation_paths]
        images, annotations = self.preprocess_batch(images), self.preprocess_batch(annotations)
        # we still need to aggregate the images into a single tensor:
        images = torch.stack(images).to(self.cfg.device)
        annotations = torch.stack(annotations).to(self.cfg.device)
        return images, annotations

    def __getitem__(self, val):
        if isinstance(val, int):
            return self.get_item_by_index(val)
        elif isinstance(val, slice):
            return self.get_item_by_slice(val)
        raise NotImplementedError(f'only able to provide items by index or by slice, not with argument {val}')

    def __len__(self):
        return len(self.image_names)

    def get_image_names(self):
        images_dir = os.path.join(self.cfg.data.root, self.cfg.data[self._split].image_dir)
        image_names = [file_name.split('.')[0] for file_name in os.listdir(images_dir)]
        return image_names

    def get_class_names(self):
        raise NotImplementedError()