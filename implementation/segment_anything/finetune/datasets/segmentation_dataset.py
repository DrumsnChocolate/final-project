import os

import cv2
import numpy as np
import torch
from prodict import Prodict
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as F, InterpolationMode
from torch.utils.data import Dataset

from finetune.configs.config_validation import _is_tuple_of


def ensure_image_rgb(image):
    # necessary in case the image is not yet represented as RGB.
    # arguably, there are different ways of converting grayscale to RGB, but this is the simplest
    if image.shape[0] == 3:
        return image
    return image.repeat(3, 1, 1)


def _clahe(image, clip_limit=40.0, tile_grid_size=(8, 8)):
    # taken from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/image/photometric.html#clahe
    # assumes that the image has a single channel
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Args:
        img (torch.Tensor): Image to be processed.
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).

    Returns:
        torch.Tensor: The processed image.
    """
    assert isinstance(image, torch.Tensor), 'image should be a torch.Tensor'
    image = np.array(image, dtype=np.uint8)
    assert image.ndim == 2, 'image should be HxW'
    assert isinstance(clip_limit, (float, int)), 'clip_limit should be float or int'
    assert _is_tuple_of(tile_grid_size, int), 'tile_grid_size should be a tuple of ints'
    assert len(tile_grid_size) == 2, 'tile_grid_size should be length 2'

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image = clahe.apply(image)
    return torch.Tensor(image)



class SegmentationMaskDataset(Dataset):

    def __init__(self, cfg: Prodict, split: str, image_names=None):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' not supported for {cfg.data.name} dataset"
        self.cfg = cfg
        self._split = split
        self.image_names = image_names
        if self.image_names is None:
            self.image_names = self.get_image_names()
        self.masks_per_image, self.index_to_name, self.index_to_class = self.count_masks_per_image()

    def get_image_path(self, image_name):
        return os.path.join(self.cfg.data.root, self.cfg.data[self._split].image_dir,
                            image_name + self.cfg.data.image_extension)

    def get_annotation_path(self, image_name):
        return os.path.join(self.cfg.data.root, self.cfg.data[self._split].annotation_dir,
                            image_name + self.cfg.data.annotation_extension)


    def _preprocess_image(self, image):
        preprocess = self.cfg.data.preprocess
        for step in preprocess:
            if step.name == 'resize':
                mode = step.mode
                interpolation = InterpolationMode[mode.upper()]
                image = F.resize(image, size=step.dimensions, interpolation=interpolation)
            if step.name == 'clahe':
                image = torch.stack(
                    [_clahe(channel, clip_limit=step.clip_limit, tile_grid_size=step.tile_grid_size) for channel in image],
                    axis=0
                )
        return image

    def _preprocess_mask(self, mask):
        preprocess = self.cfg.data.preprocess
        for step in preprocess:
            if step.name == 'resize':
                assert mask.dtype == torch.int64, 'mask should be int64 for resizing'
                mode = step.mode
                interpolation = InterpolationMode[mode.upper()]
                mask = F.resize(mask, size=step.dimensions, interpolation=interpolation)
            if step.name == 'clahe':
                continue  # we do not change the mask when applying clahe
        return mask


    def preprocess(self, image, mask=False):
        if mask:
            return self._preprocess_mask(image)
        return self._preprocess_image(image)

    def preprocess_batch(self, images, mask=False):
        return [self.preprocess(image, mask=mask) for image in images]

    def get_item_by_index(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_name = self.index_to_name[index]
        class_index = self.index_to_class[index]
        image_path = self.get_image_path(image_name)
        annotation_path = self.get_annotation_path(image_name)
        image = ensure_image_rgb(read_image(image_path))
        # using ImageReadMode.GRAY to be sure, though this should be done automatically already
        annotation = read_image(annotation_path, mode=ImageReadMode.GRAY)
        # zero out the annotation when the class index is 0, because that means there's no ROI
        annotation = (annotation == class_index) * (class_index != 0) * 1
        class_index = torch.Tensor([class_index]).to(self.cfg.device)[0]
        image, annotation = self.preprocess(image).to(self.cfg.device), self.preprocess(annotation, mask=True).to(self.cfg.device)
        return image, annotation, class_index

    def get_item_by_slice(self, _slice) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of a batch of images and a batch of annotations.
        The images are stacked into a single tensor, and so are the annotations.
        The dimensions of the returned images are [B, C, H, W].
        The dimensions of the returned annotations are [B, 1, H, W].

        Arguments:
            _slice (slice): slice object that indicates which indices to obtain.
        """
        indices = list(range(*_slice.indices(len(self))))
        image_names = [self.index_to_name[index] for index in indices]
        class_indices = [self.index_to_class[index] for index in indices]
        image_paths = [self.get_image_path(image_name) for image_name in image_names]
        annotation_paths = [self.get_annotation_path(image_name) for image_name in image_names]
        images = [ensure_image_rgb(read_image(image_path)) for image_path in image_paths]
        # using ImageReadMode.GRAY to be sure, though this should be done automatically already
        annotations = [read_image(annotation_path, mode=ImageReadMode.GRAY) for annotation_path in annotation_paths]
        # zero out the annotation when the class index is 0, because that means there's no ROI
        annotations = [(annotation == class_index) * (class_index != 0) * 1 for annotation, class_index in zip(annotations, class_indices)]
        images, annotations = self.preprocess_batch(images), self.preprocess_batch(annotations, mask=True)
        # we still need to aggregate the images into a single tensor:
        class_indices = torch.Tensor(class_indices).to(self.cfg.device)
        if len(images) == 0:
            print('images:', images)
            print('image names:', image_names)
            print('slice:', _slice)
            print('indices:', _slice.indices(len(self)))
            print('length self:', len(self))
        images = torch.stack(images).to(self.cfg.device)
        annotations = torch.stack(annotations).to(self.cfg.device)
        return images, annotations, class_indices

    def __getitem__(self, val) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(val, int):
            return self.get_item_by_index(val)
        elif isinstance(val, slice):
            return self.get_item_by_slice(val)
        raise NotImplementedError(f'only able to provide items by index or by slice, not with argument {val}')

    def __len__(self):
        return sum(self.masks_per_image.values())

    def get_image_names(self):
        images_dir = os.path.join(self.cfg.data.root, self.cfg.data[self._split].image_dir)
        image_names = ['.'.join(file_name.split('.')[:-1]) for file_name in os.listdir(images_dir)]
        return image_names

    def count_masks_per_image(self):
        masks_per_image = {}
        index_to_name = {}
        index_to_class = {}
        lower_bound = 0
        for image_name in self.image_names:
            annotation_path = self.get_annotation_path(image_name)
            annotation = read_image(annotation_path, mode=ImageReadMode.GRAY)
            if torch.all(annotation == 0):  # there is only background in the annotation
                masks_per_image[image_name] = 1
                upper_bound = lower_bound + 1
                index_to_name[lower_bound] = image_name
                index_to_class[lower_bound] = 0
                lower_bound = upper_bound
                continue
            unique_classes = torch.unique(annotation)
            class_count = unique_classes.shape[0]
            masks_per_image[image_name] = class_count - 1  # - 1 for background
            upper_bound = lower_bound + masks_per_image[image_name]
            for index in range(lower_bound, upper_bound):
                index_to_name[index] = image_name
                index_to_class[index] = unique_classes[index - lower_bound + 1]  # + 1 for background
            lower_bound = upper_bound
        assert lower_bound == sum(masks_per_image.values())
        return masks_per_image, index_to_name, index_to_class


    def get_class_names(self):
        raise NotImplementedError()