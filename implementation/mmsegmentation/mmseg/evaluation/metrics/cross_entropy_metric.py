from typing import Optional, Sequence

import torch

from mmengine.utils import mkdir_or_exist
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

@METRICS.register_module()
class CrossEntropyMetric(BaseMetric):
    """Cross Entropy evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255
        output_dir (str, optional): The directory for output prediction. Defaults to None.
        prefix (str, optional): The prefix that will be added in the metric names to disambiguate homonymous metrics
            of different evaluators. If prefix is not provided in the argument, self.default_prefix will be used
            instead. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        pos_weight (int, optional): The relative weight of the positive class.
            If not specified, will use uniform weight. Default: None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 pos_weight=None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.pos_weight = pos_weight

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): The input data batch.
            data_samples (Sequence[dict]): The data samples.
        """
        num_classes = len(self.dataset_meta['classes'])
        assert num_classes == 1, 'CrossEntropyMetric only supports binary segmentation for now.'
        for data_sample in data_samples:
            pred_logits = data_sample['seg_logits']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_logits)
            self.results.append(self._calculate(pred_logits, label, self.ignore_index))

    def _calculate(self, pred_logits: torch.tensor, label: torch.tensor, ignore_index: int):
        assert pred_logits.dim() == label.dim()
        # first, we need to change the label to 0,1 encoding. 1 for foreground
        label[label == 0] = 1
        label[label == ignore_index] = 0
        cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, label.float(),
                                                                             pos_weight=self.pos_weight,
                                                                             reduction='mean')
        return cross_entropy

    def compute_metrics(self, results: list) -> dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of the metrics, and the values are the
                corresponding results. The keys include for now only 'mCE'.
        """
        cross_entropies = results
        mean_cross_entropy = sum(cross_entropies) / len(cross_entropies)

        return {'mCE': mean_cross_entropy}
