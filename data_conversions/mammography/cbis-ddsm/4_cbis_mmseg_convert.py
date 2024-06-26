import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

mammo_root = os.path.join(os.getcwd(), "data", "mammography")
cbis_root = os.path.join(mammo_root, "cbis-ddsm")
cbis_linked_root = os.path.join(mammo_root, "cbis-linked")
mmseg_data_root = os.path.join(os.getcwd(), "implementation", "mmsegmentation", "data")


def symlink_sample(sample):
    # link both binary class mask and multi class mask
    split = sample['Split'].lower()
    img_path = os.path.join(cbis_root, 'reprocessed_data_8bit', sample['ShortPath'])
    binary_mask_path = os.path.join(cbis_root, 'merged-binary-cropped-roi-images',
                                    sample['ImageName'] + "_binary_mask.png")
    multi_mask_path = os.path.join(cbis_root, 'merged-multi-cropped-roi-images',
                                   sample['ImageName'] + "_multi_mask.png")
    mass_binary_mask_path  = os.path.join(cbis_root, 'merged-mass-binary-cropped-roi-images', sample['ImageName'] + "_binary_mask.png")

    img_link_path = os.path.join(cbis_linked_root, 'images', split, sample['ImageName'] + '.png')
    binary_mask_link_path = os.path.join(cbis_linked_root, 'annotations_binary', split, sample['ImageName'] + '.png')
    multi_mask_link_path = os.path.join(cbis_linked_root, 'annotations_multi', split, sample['ImageName'] + '.png')

    os.makedirs(os.path.dirname(img_link_path), exist_ok=True)
    os.makedirs(os.path.dirname(binary_mask_link_path), exist_ok=True)
    os.makedirs(os.path.dirname(multi_mask_link_path), exist_ok=True)

    os.symlink(img_path, img_link_path)
    os.symlink(binary_mask_path, binary_mask_link_path)
    os.symlink(multi_mask_path, multi_mask_link_path)
    if not os.path.exists(mass_binary_mask_path):
        return

    mass_img_link_path = os.path.join(cbis_linked_root, 'images_mass', split, sample['ImageName'] + '.png')
    os.makedirs(os.path.dirname(mass_img_link_path), exist_ok=True)
    os.symlink(img_path, mass_img_link_path)
    mass_binary_mask_link_path = os.path.join(cbis_linked_root, 'annotations_mass_binary', split, sample['ImageName'] + '.png')
    os.makedirs(os.path.dirname(mass_binary_mask_link_path), exist_ok=True)
    os.symlink(mass_binary_mask_path, mass_binary_mask_link_path)


def main():
    # we do the same type of merging as in cbis_mask_alignment.py, but without the crop df
    samples_df = pd.read_csv(os.path.join(cbis_root, "cbis-ddsm_singleinstance_groundtruth.csv"), sep=';')

    # create a column that indicates the split for each sample
    conditions = [
        samples_df['ImageName'].str.contains("Test"),
        samples_df['ImageName'].str.contains("Train")
    ]
    choices = ['Test', 'Train']
    samples_df['Split'] = np.select(conditions, choices)

    test_df = samples_df[samples_df["ImageName"].str.contains("Test")]
    train_df = samples_df[samples_df["ImageName"].str.contains("Train")]
    train_indices = samples_df[samples_df["Split"] == 'Train'].index
    # the exact same split used for vpt
    train_split_indices, val_split_indices = train_test_split(train_indices, test_size=0.1, shuffle=True, random_state=218, stratify=train_df["ImageLabel"])

    # we need to do some pandas gymnastics to apply this same split to the mask df:
    samples_df.loc[list(val_split_indices), 'Split'] = 'Val'

    samples_df.apply(symlink_sample, axis=1)
    # finally, we symlink from mmseg data dir to the newly created cbis-linked dir
    mmseg_cbis_linked_dir = os.path.join(mmseg_data_root, 'cbis', 'cbis-linked')
    if os.path.exists(mmseg_cbis_linked_dir):
        os.unlink(mmseg_cbis_linked_dir)
    else:
        os.makedirs(os.path.dirname(mmseg_cbis_linked_dir), exist_ok=True)
    os.symlink(cbis_linked_root, mmseg_cbis_linked_dir)


if __name__ == '__main__':
    main()
