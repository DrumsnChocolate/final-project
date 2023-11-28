import cv2
import pandas as pd
import os

vindr_root = os.path.join(os.getcwd(), "mammography", 'vindr')


def convert_16_to_8(sample):
    img16_path = os.path.join(vindr_root, 'processed_png_16bit', sample['ShortPath'])
    img16 = cv2.imread(img16_path, cv2.IMREAD_UNCHANGED)
    img8_path = os.path.join(vindr_root, 'reprocessed_png_8bit', sample['ShortPath'])
    os.makedirs(os.path.dirname(img8_path), exist_ok=True)
    img8 = (img16/256).astype('uint8')
    cv2.imwrite(img8_path, img8)

def main():
    samples_df_path = os.path.join(vindr_root, 'MG_training_files_vindr_singleinstance_groundtruth.csv')
    samples_df = pd.read_csv(samples_df_path, sep=';')
    samples_df.apply(convert_16_to_8, axis=1)


if __name__ == '__main__':
    main()
