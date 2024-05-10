from typing import Callable, Any

import numpy as np
import torch

from finetune.models.sam_wrapper import SamWrapper


def iou_metric_best_mask(mask_logits: torch.Tensor, predictions: torch.Tensor, target: torch.Tensor, iou_predictions: torch.Tensor):
    """
    Intersection over Union (IoU) Expects each tensor to be of shape BxMxHxW,
    except iou_predictions, which is of shape BxM.
    M stands for the separate mask dimension

    :param mask_logits: unnormed mask logits
    :param predictions: predicted masks, values either 0 or 1
    :param target: target masks, values either 0 or 1
    :param iou_predictions: predicted values for the IoU of each mask.
        This can be used to select the most confident mask for each sample.

    :return: for each sample, the IoU for the most confident mask. Averaged over the batch. Shape is 1.
    """
    eps = 1e-7
    areas_intersection = torch.sum(predictions * target, axis=(2,3))
    areas_union = torch.sum((predictions + target) > 0, axis=(2,3))
    # iou_per_mask has shape BxM
    iou_per_mask = ((areas_intersection + eps) / (areas_union + eps))
    best_mask_indices = iou_predictions.argmax(axis=1, keepdim=True)
    best_mask_selector = torch.zeros_like(iou_predictions).scatter(dim=1, index=best_mask_indices, value=1)
    best_mask_ious = torch.masked_select(iou_per_mask, best_mask_selector == 1)
    return best_mask_ious.mean(axis=0)


def iou_metric_per_mask(mask_logits: torch.Tensor, predictions: torch.Tensor, target: torch.Tensor, iou_predictions: torch.Tensor):
    """
    Intersection over Union (IoU) Expects each tensor to be of shape BxMxHxW,
    except iou_predictions, which is of shape BxM.
    M stands for the separate mask dimension

    :param mask_logits: unnormed mask logits
    :param predictions: predicted masks, values either 0 or 1
    :param target: target masks, values either 0 or 1
    :param iou_predictions: predicted values for the IoU of each mask.
        This can be used to select the most confident mask for each sample.

    :return: the area of the intersection, divided by the area of the union, averaged over the batch. Shape is M.
    """
    eps = 1e-7
    areas_intersection = torch.sum(predictions * target, axis=(2,3))
    areas_union = torch.sum((predictions + target) > 0, axis=(2,3))
    iou = ((areas_intersection + eps) / (areas_union + eps)).mean(axis=0)
    return iou


def dice_metric_best_mask(mask_logits: torch.Tensor, predictions: torch.Tensor, target: torch.Tensor, iou_predictions: torch.Tensor):
    """
    Intersection over Union (IoU) Expects each tensor to be of shape BxMxHxW,
    except iou_predictions, which is of shape BxM.
    M stands for the separate mask dimension

    :param mask_logits: unnormed mask logits
    :param predictions: predicted masks, values either 0 or 1
    :param target: target masks, values either 0 or 1
    :param iou_predictions: predicted values for the IoU of each mask.
        This can be used to select the most confident mask for each sample.

    :return: for each sample, the dice coefficient for the most confident mask. Averaged over the batch. Shape is 1.
    """
    eps = 1e-7
    areas_intersection = torch.sum(predictions * target, axis=(2,3))
    areas_prediction = torch.sum(predictions, axis=(2,3))
    areas_target = torch.sum(target, axis=(2,3))
    # dice_per_mask has shape BxM
    dice_per_mask = ((2 * areas_intersection + eps) / (areas_prediction + areas_target + eps))
    best_mask_indices = iou_predictions.argmax(axis=1, keepdim=True)
    best_mask_selector = torch.zeros_like(iou_predictions).scatter(dim=1, index=best_mask_indices, value=1)
    best_mask_dices = torch.masked_select(dice_per_mask, best_mask_selector == 1)
    return best_mask_dices.mean(axis=0)


def dice_metric_per_mask(mask_logits: torch.Tensor, predictions: torch.Tensor, target: torch.Tensor, iou_predictions: torch.Tensor):
    """
    Dice Coefficient. Expects each tensor to be of shape BxMxHxW,
    except iou_predictions, which is of shape BxM.
    M stands for the separate mask dimension

    :param mask_logits: unnormed mask logits
    :param predictions: predicted masks, values either 0 or 1
    :param target: target masks, values either 0 or 1
    :param iou_predictions: predicted values for the IoU of each mask.
        This can be used to select the most confident mask for each sample.

    :return: the dice coefficient, averaged over the batch. Shape is M.
    """
    eps = 1e-7
    areas_intersection = torch.sum(predictions * target, axis=(2,3))
    areas_prediction = torch.sum(predictions, axis=(2,3))
    areas_target = torch.sum(target, axis=(2,3))
    dice = ((2 * areas_intersection + eps) / (areas_prediction + areas_target + eps)).mean(axis=0)
    return dice


def choose_metric_function(metric_definition):
    if not metric_definition.per_mask:
        if metric_definition.name == 'IoU':
            return iou_metric_best_mask
        if metric_definition.name == 'Dice':
            return dice_metric_best_mask

    if metric_definition.name == 'IoU':
        return iou_metric_per_mask
    if metric_definition.name == 'Dice':
        return dice_metric_per_mask
    return None


def build_metric_function(metric_definition) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    metric = choose_metric_function(metric_definition)
    if metric is None:
        raise NotImplementedError()
    return lambda *args: metric(*args).tolist()


def build_metric_name(metric_definition):
    if not metric_definition.per_mask:
        return metric_definition.name
    return f'per_mask_{metric_definition.name}'


def build_metric_functions(cfg):
    return {build_metric_name(metric): build_metric_function(metric) for metric in cfg.model.metrics}


def call_metrics(
        metric_functions: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        output_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_batch: torch.Tensor,
        model: SamWrapper
) -> dict[str, torch.Tensor]:
    mask_batch, iou_predictions, _ = output_batch
    prediction_batch: torch.Tensor = (mask_batch > model.mask_threshold) * 1
    target_batch = target_batch.repeat(1, mask_batch.shape[1], 1, 1)
    assert target_batch.shape == mask_batch.shape
    return {key: metric_function(mask_batch, prediction_batch, target_batch, iou_predictions) for key, metric_function in metric_functions.items()}


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
