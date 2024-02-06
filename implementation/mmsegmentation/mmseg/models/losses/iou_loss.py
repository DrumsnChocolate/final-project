from typing import Union

import torch
from .utils import weight_reduce_loss


def iou_loss(pred: torch.Tensor,
             target: torch.Tensor,
             weight: Union[torch.Tensor, None],
             eps: float = 1e-3,
             reduction: Union[str, None] = 'mean',
             avg_factor: Union[float, None] = None,
             ignore_index: int = 255) -> float:
    """Calculate IoU loss, arguments similar to dice loss."""
    if ignore_index is not None:
        num_classes = pred.shape[1]
        pred = pred[:, torch.arange(num_classes) != ignore_index, :, :]
        target = target[:, torch.arange(num_classes) != ignore_index, :, :]
        assert pred.shape[1] != 0 # if the ignored index is the only class
    input = pred.flatten(1)
    target = target.flatten(1).float()
    a = torch.sum(input * target > 0, 1)
    b = torch.sum((input > 0) + (target > 0), 1)  # boolean addition is the same as logical or
    iou = a / b
    loss = 1 - iou
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

# @Models.register_module()
# class IOULoss(nn.Module):
#
#     def __init__(self,
#                  ):