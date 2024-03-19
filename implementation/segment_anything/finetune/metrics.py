from typing import Callable, Any

import numpy as np
import torch

from models import SamWrapper


def iou_metric(mask_logits: torch.Tensor, predictions: torch.Tensor, target: torch.Tensor):
    """
    Intersection over Union (IoU) Expects each tensor to be of shape BxMxHxW

    :param mask_logits: unnormed mask logits
    :param predictions: predicted masks, values either 0 or 1
    :param target: target masks, values either 0 or 1
    :return: the area of the intersection, divided by the area of the union, averaged over the batch.
    """
    eps = 1e-7
    areas_intersection = torch.sum(predictions * target, axis=(2,3))
    areas_union = torch.sum((predictions + target) > 0, axis=(2,3))
    iou = (areas_intersection / (areas_union + eps)).mean(axis=0)
    return iou

def dice_metric(mask_logits: torch.Tensor, predictions: torch.Tensor, target: torch.Tensor):
    """
    Dice Coefficient. Expects each tensor to be of shape BxMxHxW, M for the separate mask dimension

    :param mask_logits: unnormed mask logits
    :param predictions: predicted masks, values either 0 or 1
    :param target: target masks, values either 0 or 1

    :return: the dice coefficient, averaged over the batch.
    """
    eps = 1e-7
    areas_intersection = torch.sum(predictions * target, axis=(2,3))
    areas_prediction = torch.sum(predictions, axis=(2,3))
    areas_target = torch.sum(target, axis=(2,3))
    dice = (2 * areas_intersection / (areas_prediction + areas_target + eps)).mean(axis=0)
    return dice




def build_metric_function(metric_definition) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    metric = None
    if metric_definition.name == 'IoU':
        metric = iou_metric
    if metric_definition.name == 'Dice':
        metric = dice_metric
    if metric is None:
        raise NotImplementedError()
    return lambda *args: metric(*args).tolist()


def build_metric_functions(cfg):
    return {metric.name: build_metric_function(metric) for metric in cfg.model.metrics}


def call_metrics(
        metric_functions: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        output_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_batch: torch.Tensor,
        model: SamWrapper
) -> dict[str, torch.Tensor]:
    mask_batch, _, _ = output_batch
    prediction_batch: torch.Tensor = (mask_batch > model.mask_threshold) * 1
    target_batch = target_batch.repeat(1, mask_batch.shape[1], 1, 1)
    assert target_batch.shape == mask_batch.shape
    return {key: metric_function(mask_batch, prediction_batch, target_batch) for key, metric_function in metric_functions.items()}


def append_metrics(metrics: dict[str, list[Any]], new_metrics: dict[str, Any]):
    if len(metrics.keys()) == 0:
        for k, v in new_metrics.items():
            metrics[k] = [v]
        return
    assert metrics.keys() == new_metrics.keys(), f'Expected metrics to have keys {metrics.keys()}, but got {new_metrics.keys()}'
    for k, v in new_metrics.items():
        metrics[k].append(v)


def average_metrics(metrics: dict[str, list[Any]]):
    for k, v in list(metrics.items()):
        metrics[f'avg_{k}'] = np.mean(v, axis=0).tolist()
