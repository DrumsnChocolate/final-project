from mmengine.dataset import BaseDataset

from mmseg.registry import DATASETS


@DATASETS.register_module()
class CBISDataset(BaseDataset):
    """CBIS-DDSM dataset.

    In segmentation maps for CBIS-DDSM, 0 stands for background
    """
    METAINFO = dict(
        classes=("mass", "calc"),
        palette=[[255, 0, 0], [0, 0, 255]],
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
