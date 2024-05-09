from finetune.datasets.segmentation_dataset import SegmentationMaskDataset

zgt_binary_class_names = ('background', 'roi')

class ZGTBinaryMaskDataset(SegmentationMaskDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_class_names(self):
        return zgt_binary_class_names