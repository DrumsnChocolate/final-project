#!/usr/bin/env python3

"""Image transformations."""
import torchvision as tv


def get_transforms(split, size, crop=True):
    if crop:
        return get_transforms_crop(split, size)
    return get_transforms_no_crop(split, size)

def get_normalize():
    return tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

def get_transforms_no_crop(split, size):
    if split == "train":
        return tv.transforms.Compose(
            [
                tv.transforms.Resize((size, size)),
                tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(),
                get_normalize(),
            ]
        )
    return tv.transforms.Compose(
        [
            tv.transforms.Resize((size, size)),
            tv.transforms.ToTensor(),
            get_normalize(),
        ]
    )

def get_transforms_crop(split, size):
    if size == 448:
        resize_dim = 512
        crop_dim = 448
    elif size == 224:
        resize_dim = 256
        crop_dim = 224
    elif size == 384:
        resize_dim = 438
        crop_dim = 384
    elif size == 672:
        resize_dim = 768
        crop_dim = 672
    elif size == 896:
        resize_dim = 1024
        crop_dim = 896
    if split == "train":
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize((resize_dim, resize_dim)),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(),
                get_normalize(),
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize((resize_dim, resize_dim)),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                get_normalize(),
            ]
        )
    return transform
