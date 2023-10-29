import pandas as pd
from PIL import Image
import os

def crop_image(img, crop_row):
    return img.crop((crop_row["pro_min_x"], crop_row["pro_min_y"], crop_row["pro_max_x"], crop_row["pro_max_y"]))

def mask_exists(sample):
    dir_path = os.getcwd()
    raw_mask_path = os.path.join(dir_path, 'data', "mammography", "cbis-ddsm", "roi-images", sample["ImageName"][:-2], "1-2.png")
    return os.path.exists(raw_mask_path)

def load_crop_save_mask(sample):
    if not mask_exists(sample):
        return  # todo: find out why we are missing masks?
    dir_path = os.getcwd()
    raw_mask_path = os.path.join(dir_path, 'data', "mammography", "cbis-ddsm", "roi-images", sample["ImageName"][:-2], "1-2.png")
    raw_mask = Image.open(raw_mask_path)
    cropped_mask = crop_image(raw_mask, sample)
    cropped_mask_path = os.path.join(dir_path, 'data', "mammography", "cbis-ddsm", "cropped-roi-images", sample["ImageName"][:-2], "1-2.png")
    os.makedirs(os.path.dirname(cropped_mask_path), exist_ok=True)
    cropped_mask.save(cropped_mask_path)

def main():
    samples_df_path = os.path.join(os.getcwd(), 'data', 'mammography', 'cbis-ddsm', 'cbis-ddsm_singleinstance_groundtruth.csv')
    crop_df_path = os.path.join(os.getcwd(), 'data', 'mammography', 'cbis-ddsm', 'cbis-ddsm_singleinstance_imgpreprocessing_size_mod.csv')
    samples_df = pd.read_csv(samples_df_path, sep=';')
    crop_df = pd.read_csv(crop_df_path, sep=';')
    joined_df = samples_df.join(crop_df.set_index('ImageName'), on='ImageName')
    joined_df.apply(load_crop_save_mask, axis=1)


if __name__ == '__main__':
    main()
