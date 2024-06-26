from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CBISMonoDataset(BaseSegDataset):
    """CBIS-DDSM mono class dataset.
    """
    METAINFO = dict(
        classes=("roi",),
        palette=[[255, 255, 255]],  # black
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class CBISBinaryDataset(BaseSegDataset):
    """CBIS-DDSM binary class dataset.
    """
    METAINFO = dict(
        classes=("background", "roi"),
        palette=[[0, 0, 0], [255, 255, 255]],  # black, white
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class CBISMultiDataset(BaseSegDataset):
    """CBIS-DDSM multi class dataset.
    """
    METAINFO = dict(
        classes=("background", "calcification benign", "calcification malignant", "mass benign", "mass malignant"),
        palette=[[127, 127, 127], [255, 255, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]],  # grey, yellow, green, blue, black
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)



