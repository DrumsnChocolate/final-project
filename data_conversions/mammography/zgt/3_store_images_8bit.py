import cv2
from torchvision.io import write_png
import os
import pandas as pd
from os.path import pardir
import torch

cur_dir = os.getcwd()
exports_root = os.path.join(cur_dir, 'data/mammography/zgt/label_studio/veltman')
annotations_root = os.path.abspath(os.path.join(cur_dir, 'data/mammography/zgt-linked'))
storage_root = os.path.abspath(os.path.join(cur_dir, 'data/mammography/zgt/remote_images'))


def read_and_store_image(row):
    # loads image in CxHxW for color, HxW for grayscale
    image = cv2.imread(os.path.join(storage_root, row['image_path']), cv2.IMREAD_GRAYSCALE)
    # convert to tensor with 1xHxW
    image = torch.tensor(image, dtype=torch.uint8).unsqueeze(0)
    # write_png requires HxWxC for color, 1xHxW for grayscale
    write_png(image, os.path.join(annotations_root, 'images', row['image_name']), compression_level=0)

def main():
    with open(os.path.join(exports_root, 'parsed_exports.csv')) as f:
        annotations = pd.read_csv(f)

    os.makedirs(os.path.join(annotations_root, 'images'), exist_ok=True)
    annotations.apply(read_and_store_image, axis=1)



if __name__ == '__main__':
    main()