from finetune.datasets.segmentation_dataset import SegmentationDataset, SegmentationMaskDataset

cbis_binary_class_names = ('background', 'roi')
cbis_multi_class_names = ('background', 'calcification benign', 'calcification malignant', 'mass benign', 'mass malignant')


# the cbis binary dataset and the cbis binary mask dataset
# are practically equivalent, but define both for convention
class CBISBinaryDataset(SegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_class_names(self):
        return cbis_binary_class_names


class CBISBinaryMaskDataset(SegmentationMaskDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_class_names(self):
        return cbis_binary_class_names


class CBISMultiDataset(SegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_class_names(self):
        return cbis_multi_class_names


class CBISMultiMaskDataset(SegmentationMaskDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_class_names(self):
        return cbis_multi_class_names