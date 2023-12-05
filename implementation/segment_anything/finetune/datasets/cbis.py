from finetune.datasets.segmentation_dataset import SegmentationDataset


class CBISBinaryDataset(SegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_class_names(self):
        return ('background', 'roi')


class CBISMultiDataset(SegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_class_names(self):
        return ('background', 'calcification benign', 'calcification malignant', 'mass benign', 'mass malignant')