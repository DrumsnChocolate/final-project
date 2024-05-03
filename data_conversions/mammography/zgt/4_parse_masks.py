import pandas as pd
import ast
import cv2
import numpy as np
import os
import scipy

cur_dir = os.getcwd()
exports_root = os.path.join(cur_dir, 'data/mammography/zgt/label_studio/veltman')
annotations_root = os.path.join(cur_dir, 'data/mammography/zgt-linked')

# taken from https://stackoverflow.com/questions/74339154/how-to-convert-rle-format-of-label-studio-to-black-and-white-image-masks


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def region_to_mask(region: dict) -> np.array:
    """
    Converts rle to image mask
    Args:
        region: region with properties rle, height, and width.

    Returns: np.array
    """
    rle = region['rle']
    if rle == []:
        return np.array([])
    height = int(region['height'])
    width = int(region['width'])
    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image


def replace_stub(row):
    if len(row['mask']) != 0:
        return row['mask']
    image_path = os.path.join(annotations_root, 'images', row['image_name'])
    image = cv2.imread(image_path)
    return np.zeros(image.shape[:-1])


def fill_holes(mask):
    return scipy.ndimage.binary_fill_holes(mask, structure=np.ones((3,3))) * 1


def merge_rois_single_class(masks):
    masks = np.array(masks)
    return (masks.sum(axis=0) > 0) * 1


def merge_rois_multi_class(masks):
    masks = np.array(masks)
    merged_mask = np.zeros(masks[0].shape)
    for i, mask in enumerate(masks):
        merged_mask += mask * (i + 1)
    return merged_mask


def store_single_class(row):
    mask_path = os.path.join(annotations_root, 'annotations_binary', row['image_name'])
    mask = row['mask'] * 50
    cv2.imwrite(mask_path, mask)


def store_multi_class(row):
    mask_path = os.path.join(annotations_root, 'annotations_multi', row['image_name'])
    mask = row['mask'] * 50
    cv2.imwrite(mask_path, mask)


def main():
    df = pd.read_csv(os.path.join(exports_root, 'parsed_exports.csv'), converters={'rle': ast.literal_eval})
    df['mask'] = df.apply(region_to_mask, axis=1)
    df['mask'] = df.apply(replace_stub, axis=1)
    df['mask'] = df['mask'].apply(fill_holes)

    os.makedirs(os.path.join(annotations_root, 'annotations_binary'), exist_ok=True)
    os.makedirs(os.path.join(annotations_root, 'annotations_multi'), exist_ok=True)

    df_binary_masks = df.groupby('image_name', dropna=False).agg({'mask': merge_rois_single_class}).reset_index()
    df_multi_masks = df.groupby('image_name', dropna=False).agg({'mask': merge_rois_multi_class}).reset_index()

    df_binary_masks.apply(store_single_class, axis=1)
    df_multi_masks.apply(store_multi_class, axis=1)







if __name__ == '__main__':
    main()