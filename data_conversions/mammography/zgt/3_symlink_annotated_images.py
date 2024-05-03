import pandas as pd
import os

cur_dir = os.getcwd()
exports_root = os.path.join(cur_dir, 'data/mammography/zgt/label_studio/veltman')
annotations_root = os.path.join(cur_dir, 'data/mammography/zgt-linked')
storage_root = os.path.join(cur_dir, 'data/mammography/zgt/remote_images')



def symlink_row(row):

    link_path = os.path.join(annotations_root, 'images', row['image_name'])
    target = os.path.join(storage_root, row['image_path'])
    if not os.path.exists(link_path):
        os.symlink(target, link_path)


def main():
    with open(os.path.join(exports_root, 'parsed_exports.csv')) as f:
        annotations = pd.read_csv(f)

    os.makedirs(os.path.join(annotations_root, 'images'), exist_ok=True)
    annotations.apply(symlink_row, axis=1)



if __name__ == '__main__':
    main()
