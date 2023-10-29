import os
import pandas as pd


def symlink_sample(sample):
    dir_path = os.getcwd()
    converted_path = os.path.join(dir_path, 'data', 'mammography', "cbis-linked")
    img_path = os.path.join(dir_path, 'data', 'mammography', 'cbis-ddsm', 'reprocessed_data_8bit', sample['ShortPath'])
    mask_path = os.path.join(dir_path, 'data', 'mammography', 'cbis-ddsm', 'cropped-roi-images', sample['ImageName'][:-2],
                             "1-2.png")
    if not os.path.isfile(mask_path):
        return  # todo: we are missing masks, which is why this happens.
    split = 'nosplit'
    if 'Training' in sample['ImageName']:
        split = 'train'
    if 'Test' in sample['ImageName']:
        split = 'test'
    img_link_path = os.path.join(converted_path, 'images', split, sample['ImageName'] + '.png')
    mask_link_path = os.path.join(converted_path, 'annotations', split, sample['ImageName'] + '.png')
    os.makedirs(os.path.dirname(img_link_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_link_path), exist_ok=True)
    os.symlink(img_path, img_link_path)
    os.symlink(mask_path, mask_link_path)


def main():
    dir_path = os.getcwd()
    samples_df_path = os.path.join(dir_path, 'data', 'mammography', 'cbis-ddsm', 'cbis-ddsm_singleinstance_groundtruth.csv')
    samples_df = pd.read_csv(samples_df_path, sep=';')
    samples_df.apply(symlink_sample, axis=1)


if __name__ == '__main__':
    main()
