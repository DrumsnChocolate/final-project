from mmengine.dataset import BaseDataset

from mmseg.registry import DATASETS


@DATASETS.register_module()
class CBISDataset(BaseDataset):
    """CBIS-DDSM dataset.
    """
    METAINFO = dict(
        classes=("background", "roi"),
        palette=[[255, 0, 0], [0, 0, 255]],
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
