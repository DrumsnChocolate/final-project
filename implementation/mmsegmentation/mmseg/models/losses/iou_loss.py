from typing import Union

import torch
from torch import nn

from .dice_loss import _expand_onehot_labels_dice
from .utils import weight_reduce_loss
from ...registry import MODELS


def iou_loss(pred: torch.Tensor,
             target: torch.Tensor,
             weight: Union[torch.Tensor, None],
             eps: float = 1e-7,
             reduction: Union[str, None] = 'mean',
             naive=False,
             signed=False,
             avg_factor: Union[float, None] = None,
             ignore_index: int = 255) -> float:
    """Calculate IoU loss, arguments similar to dice loss.
    The precise definition used here for 'naive' is the one from connected-unets:
    https://github.com/AsmaBaccouche/Connected-Unets-and-more/blob/main/Basic_Unet_segmnetation.py

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive (bool, optional): If true, use the iou
            loss defined in the Connect-UNets repository, otherwise, use the
            iou loss in which the magnitudes of the numerator and denominator
            correspond better to the binary definition of Intersection over Union.
             Default: False.
        signed (bool, optional): Mutually exclusive with naive. If true, preserve
            the sign of the product in the intersection, to prevent the loss from
            rewarding when x=-y. Default: False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to 255.

    """

    assert not naive or not signed, "naive and signed are mutually exclusive"
    if ignore_index is not None:
        num_classes = pred.shape[1]
        pred = pred[:, torch.arange(num_classes) != ignore_index, :, :]
        target = target[:, torch.arange(num_classes) != ignore_index, :, :]
        assert pred.shape[1] != 0  # if the ignored index is the only class
    input = pred.flatten(1)
    target = target.flatten(1).to(input)
    if naive:
        intersection = torch.sum(torch.abs(input*target), 1)
        union = torch.sum(input, 1) + torch.sum(target, 1) - intersection
        iou = (intersection + eps) / (union + eps)
    else:
        # we divert from the Connected U-Nets implementation here,
        # because we want to ensure that the denominator > numerator,
        # as long as the domain of the function falls within the real numbers.
        # Even more importantly, we want to ensure that the denominator > 0.
        # This is because we both want to avoid division by zero, and a bifurcation
        # in the loss function, where the loss would suddenly become greater than 1
        # if the denominator becomes negative.
        # Connected U-Nets uses (loosely speaking)
        # |A*B| / (A + B - |A*B|)
        # while we prefer
        # sqrt(|A*B|) / (|A| + |B| - sqrt(|A*B|))
        # As it is closer to the definition of Intersection over Union.
        # Specifically, this allows us to mathematically prove that both our
        # requirements (denominator > 0, and denominator >= numerator) are met
        # by the use of this loss function.

        # eps for keeping the derivative of the sqrt from being infinite.
        # This is a very small number, so it should not destabilize the loss.
        # it's much smaller than regular eps, because we add it to each element of a matrix.
        sqrt_eps = torch.tensor(1e-20)
        if signed:
            # preserves the sign of the product, by taking
            # Intersection = A*B / (sqrt(|A*B|)) instead of
            # Intersection = sqrt(|A*B|)
            # this penalizes when a=-b instead of rewarding it like a=b.
            product = input * target
            intersection = torch.sum(product/torch.sqrt(torch.abs(input*target) + sqrt_eps), 1)
        else:
            intersection = torch.sum(torch.sqrt(torch.abs(input*target) + sqrt_eps), 1)
        union = torch.sum(torch.abs(input), 1) + torch.sum(torch.abs(target), 1) - intersection
        iou = (intersection + eps) / (union + eps)
    loss = 1 - iou
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class IoULoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive=False,
                 signed=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-7,
                 loss_name='loss_iou'):
        """Compute IoU loss.

        Args similar to those of DiceLoss
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.naive = naive
        self.signed = signed
        self.loss_weight = loss_weight
        self.eps = eps
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        one_hot_target= target
        if (pred.shape != target.shape):
            one_hot_target = _expand_onehot_labels_dice(pred, target)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override is not None else self.reduction)
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                # softmax does not work when there is only 1 class
                pred = pred.softmax(dim=1)
        loss = self.loss_weight * iou_loss(
            pred,
            one_hot_target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive=self.naive,
            signed=self.signed,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
