import pandas as pd
from PIL import Image
import os

cbis_root = os.path.join(os.getcwd(), "data", "mammography", "")


def count_masks(row):
    roi_root = os.path.join(cbis_root, "roi-images")
    mask_folder = os.path.join(roi_root, row["MaskFolderName"])
    return len(os.listdir(mask_folder)) - 1  # subtract 1 because there is also the picture in there


def crop_image(img, crop_row):
    return img.crop((crop_row["pro_min_x"], crop_row["pro_min_y"], crop_row["pro_max_x"], crop_row["pro_max_y"]))


def load_crop_save_mask(sample):
    roi_root = os.path.join(cbis_root, "roi-images")
    cropped_roi_root = os.path.join(cbis_root, "cropped-roi-images")
    raw_mask_path = os.path.join(roi_root, sample["MaskFolderName"], "1-2.png")
    cropped_mask_path = os.path.join(cropped_roi_root, sample["MaskFolderName"], "1-2.png")
    raw_mask = Image.open(raw_mask_path)
    cropped_mask = crop_image(raw_mask, sample)
    os.makedirs(os.path.dirname(cropped_mask_path), exist_ok=True)
    cropped_mask.save(cropped_mask_path)

def main():
    samples_df = pd.read_csv(os.path.join(cbis_root, "cbis-ddsm_singleinstance_groundtruth.csv"), sep=';')
    crop_df = pd.read_csv(os.path.join(cbis_root, "cbis-ddsm_singleinstance_imgpreprocessing_size_mod.csv"), sep=';')
    mask_df = pd.read_csv(os.path.join(cbis_root, "MG_training_files_cbis-ddsm_roi_groundtruth.csv"), sep=';')

    columns = ['Patient_Id', 'Views', 'AbnormalityType']
    assert len(samples_df[columns].drop_duplicates()) == len(samples_df)  # assert that these columns form a unique key
    assert len(mask_df[columns].drop_duplicates()) != len(
        mask_df)  # assert that these columns do not form a unique  key for masks, since there can be multiple masks per sample
    assert len(mask_df[columns].drop_duplicates()) == len(
        samples_df)  # assert that there are as many unique value combos in mask_df as there are in samples_df
    assert len(samples_df[columns].drop_duplicates().set_index(columns).join(
        mask_df[columns].drop_duplicates().set_index(columns))) == len(
        samples_df)  # assert that for each sample there is at least one mask

    # we will be matching masks to samples using a multiindex on these columns,
    # while we will be matching samples to crops using the ImageName column

    # only use the relevant info from the mask dataframe
    mask_df = mask_df[[*columns, "FolderName"]].rename(columns={"FolderName": "MaskFolderName"})
    joined_df = mask_df.join(samples_df.set_index(columns), on=columns)
    joined_df = joined_df.join(crop_df.set_index("ImageName"), on="ImageName")
    assert len(mask_df) == len(joined_df)

    # investigate the amount of masks per folder
    joined_df["MaskCount"] = joined_df.apply(count_masks, axis=1)
    assert joined_df["MaskCount"].unique() == [1]  # there should only one mask per roi folder

    joined_df.apply(load_crop_save_mask, axis=1)


if __name__ == '__main__':
    main()
