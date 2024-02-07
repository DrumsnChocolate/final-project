from typing import Callable, Any

import numpy as np
import torch
import torchvision


def dice_single(output, target):
    # invert because we use it as a loss, not an objective
    return 2 * torch.sum(output * target) / (torch.sum(output * output) + torch.sum(target * target))


def dice_item(outputs, targets):
    return torch.vmap(dice_single)(outputs, targets)


def dice(output_batch, target_batch) -> torch.Tensor:
    assert output_batch.shape[1] in [1,
                                     3], f'Expected outputs to have 1 or 3 masks, but got {output_batch.shape[1]} masks'
    assert target_batch.shape[1] == output_batch.shape[
        1], f'Expected targets to have {output_batch.shape[1]} masks, but got {target_batch.shape[1]} masks'
    return torch.vmap(dice_item)(output_batch, target_batch)

def focal_single(output, target, alpha, gamma, reduction):
    return torchvision.ops.sigmoid_focal_loss(output, target, alpha=alpha, gamma=gamma, reduction=reduction)

def focal_item(outputs, targets, alpha, gamma, reduction):
    partial_focal_single = lambda output, target: focal_single(output, target, alpha, gamma, reduction)
    return torch.vmap(partial_focal_single)(outputs, targets)

def focal(output_batch, target_batch, alpha, gamma, reduction) -> torch.Tensor:
    partial_focal_item = lambda outputs, targets: focal_item(outputs, targets, alpha, gamma, reduction)
    return torch.vmap(partial_focal_item)(output_batch, target_batch)


def iou_single(output, target):
    intersection = torch.sum(output * target > 0)
    union = torch.sum((output > 0) + (target > 0))  # boolean addition is the same as logical or
    return intersection / union


def iou_item(outputs, targets):
    return torch.vmap(iou_single)(outputs, targets)


def iou(output_batch, target_batch) -> torch.Tensor:
    assert output_batch.shape[1] in [1,
                                     3], f'Expected outputs to have 1 or 3 masks, but got {output_batch.shape[1]} masks'
    assert target_batch.shape[1] == output_batch.shape[
        1], f'Expected targets to have {output_batch.shape[1]} masks, but got {target_batch.shape[1]} masks'

    return torch.vmap(iou_item)(output_batch, target_batch)


def mse(predicted_batch, target_batch):
    return torch.nn.functional.mse_loss(predicted_batch, target_batch, reduction='none')



def build_metric_function(metric_definition) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    metric = None
    if metric_definition.name == 'IoU':
        metric = iou
    if metric_definition.name == 'Dice':
        metric = dice
    if metric_definition.name == 'Focal':
        alpha = metric_definition.get('alpha', -1)
        gamma = metric_definition.get('gamma', 2.0)
        reduction = metric_definition.get('reduction', 'mean')
        metric = lambda outputs, targets: focal(outputs, targets.float(), alpha, gamma, reduction)
    if metric is None:
        raise NotImplementedError()
    return lambda *args: metric(*args).tolist()

def build_metric_functions(cfg):
    return {metric.name: build_metric_function(metric) for metric in cfg.model.metrics}




def call_metrics(
        metric_functions: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        output_batch: torch.Tensor,
        target_batch: torch.Tensor
) -> dict[str, torch.Tensor]:
    mask_batch, _, _ = output_batch
    target_batch = target_batch.repeat(1, mask_batch.shape[1], 1, 1)
    assert target_batch.shape == mask_batch.shape
    return {key: metric_function(mask_batch, target_batch) for key, metric_function in metric_functions.items()}



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
