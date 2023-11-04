import pandas as pd
import os
import numpy as np
import cv2

cbis_root = os.path.join(os.getcwd(), "data", "mammography", "cbis-ddsm")

cbis_merged_single_path = os.path.join(cbis_root, "merged-binary-cropped-roi-images")
cbis_merged_multi_path = os.path.join(cbis_root, "merged-multi-cropped-roi-images")
multi_label_class_ids = None


# first let's do the simple case:
def merge_rois_single_class(sample):
    mask_folder_names = sample['MaskFolderNames']
    mask_paths = [os.path.join(cbis_root, 'cropped-roi-images', mask_folder_name, '1-2.png') for mask_folder_name in
                  mask_folder_names]
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths]
    binary_masks = [(mask > 0) * 1 for mask in masks]
    merged_mask = np.zeros(binary_masks[0].shape)
    for binary_mask in binary_masks:
        merged_mask = np.maximum(merged_mask, binary_mask)
    return merged_mask


def has_overlapping_masks(sample):
    abnormality_types = sample['AbnormalityTypes']
    image_labels = sample['ImageLabels']
    mask_folder_names = sample['MaskFolderNames']
    if len(mask_folder_names) == 1:
        return False
    mask_paths = [os.path.join(cbis_root, 'cropped-roi-images', mask_folder_name, '1-2.png') for mask_folder_name in
                  mask_folder_names]
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths]
    binary_masks = [(mask > 0) * 1 for mask in masks]
    binary_masks_sum = np.sum(binary_masks, axis=0)
    assert binary_masks_sum.shape == binary_masks[0].shape
    if np.max(binary_masks_sum) == 1:
        return False
    # okay, so for this sample turns out there's some overlap. Now we will do a more expensive operation that shows us whether there is overlap between differing classes!
    for i in range(len(binary_masks)):
        mask1 = binary_masks[i]
        abnormality_type1 = abnormality_types[i]
        image_label1 = image_labels[i]
        for j in range(i + 1, len(binary_masks)):
            mask2 = binary_masks[j]
            abnormality_type2 = abnormality_types[j]
            image_label2 = image_labels[j]
            if (abnormality_type1, image_label1) == (abnormality_type2, image_label2):
                continue
            if np.max(mask1 + mask2) > 1:
                print(sample)
                return True
    return False


def merge_rois_multi_class(sample):
    # in this function, we operate under the assumption that there are no overlapping masks from different classes
    abnormality_types = sample['AbnormalityTypes']
    image_labels = sample['ImageLabels']
    mask_folder_names = sample['MaskFolderNames']
    mask_paths = [os.path.join(cbis_root, 'cropped-roi-images', mask_folder_name, '1-2.png') for mask_folder_name in
                  mask_folder_names]
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths]
    binary_masks = [(mask > 0) * 1 for mask in masks]
    class_masks = [mask * multi_label_class_ids[(abnormality_types[i], image_labels[i])] for i, mask in
                   enumerate(binary_masks)]
    merged_mask = np.zeros(class_masks[0].shape)
    for class_mask in class_masks:
        merged_mask = np.maximum(merged_mask, class_mask)
    return merged_mask


def store_single_class(sample):
    merged_mask = merge_rois_single_class(sample)
    mask_path = os.path.join(cbis_merged_single_path, sample['ImageName'] + '_binary_mask.png')
    cv2.imwrite(mask_path, merged_mask)


def store_multi_class(sample):
    merged_multi_mask = merge_rois_multi_class(sample)
    mask_path = os.path.join(cbis_merged_multi_path, sample['ImageName'] + '_multi_mask.png')
    cv2.imwrite(mask_path, merged_multi_mask)


def main():
    os.makedirs(cbis_merged_single_path, exist_ok=True)
    os.makedirs(cbis_merged_multi_path, exist_ok=True)

    # we do the same type of merging as in cbis_mask_alignment.py, but without the crop df

    samples_df = pd.read_csv(os.path.join(cbis_root, "cbis-ddsm_singleinstance_groundtruth.csv"), sep=';')
    mask_df = pd.read_csv(os.path.join(cbis_root, "MG_training_files_cbis-ddsm_roi_groundtruth.csv"), sep=';')

    columns = ['Patient_Id', 'Views', 'AbnormalityType']

    samples_df = samples_df[[*columns, "ImageName", 'ImageLabel']]

    # determine splits indicated by the image names
    conditions = [
        samples_df['ImageName'].str.contains("Test"),
        samples_df['ImageName'].str.contains("Train")
    ]
    choices = ['Test', 'Train']
    samples_df['Split'] = np.select(conditions, choices)

    mask_df = mask_df[[*columns, "FolderName"]].rename(columns={"FolderName": "MaskFolderName"})
    joined_df = mask_df.join(samples_df.set_index(columns), on=columns)

    aggregated_df = joined_df.groupby(['ImageName', 'Patient_Id', 'Views', 'Split']).agg(
        lambda x: list(x)).reset_index().rename(
        columns={'AbnormalityType': 'AbnormalityTypes', 'MaskFolderName': 'MaskFolderNames', 'ImageLabel': 'ImageLabels'})

    unique_abnormality_types = sorted(list(set(joined_df['AbnormalityType'].to_list())))
    unique_image_labels = sorted(list(set(joined_df['ImageLabel'].to_list())))
    multi_label_classes = [(abnormality_type, image_label) for abnormality_type in unique_abnormality_types for
                           image_label in unique_image_labels]
    global multi_label_class_ids
    multi_label_class_ids = {(abnormality_type, image_label): i + 1 for i, (abnormality_type, image_label) in
                             enumerate(multi_label_classes)}

    # let's make sure there's no overlap between different classes:
    aggregated_df['HasOverlappingMasks'] = aggregated_df.apply(has_overlapping_masks, axis=1)
    assert not np.any(aggregated_df['HasOverlappingMasks'].to_list())

    aggregated_df.apply(store_single_class, axis=1)
    aggregated_df.apply(store_multi_class, axis=1)


if __name__ == '__main__':
    main()
