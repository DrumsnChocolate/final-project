from typing import Callable

import torch
from metrics import iou, dice, focal, mse


def get_loss_function(loss_definition) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if loss_definition.name == 'Dice':
        return lambda outputs, targets: dice(outputs, targets)
    if loss_definition.name == 'Focal':
        alpha = loss_definition.get('alpha', -1)  # -1 means no class balancing
        gamma = loss_definition.get('gamma', 2.0)  # 2.0 is the default value from torchvision
        # 'mean' is not normally the default value, but we want equal impact from each sample, regardless of image size.
        reduction = loss_definition.get('reduction', 'mean')
        return lambda outputs, targets: focal(outputs, targets.float(), alpha, gamma, reduction)
    raise NotImplementedError()

def build_loss_function(cfg):
    loss_parts = [get_loss_function(loss_item) for loss_item in cfg.model.loss.parts]
    loss_weights = [loss_item.weight for loss_item in cfg.model.loss.parts]
    total_weight = sum(loss_weights)
    return lambda outputs, targets: sum([weight * loss_part(outputs, targets) for weight, loss_part in zip(loss_weights, loss_parts)]) / total_weight


def find_minimum_loss(zipped_losses):
    # zipped_losses has shape [batch_size, num_masks, 2]. We are looking for the tuple with the smallest first element.
    # we can do this by sorting the second dimension, and then taking all first elements along that dimension.
    # this is a bit of a hack, but it works.
    # note that this introduces a bias toward lower iou predictions, because those are the second element in the tuple,
    # and determine tie-breaking. For example, (1,2) < (1,3).
    sorted_losses = torch.sort(zipped_losses, dim=1)
    return sorted_losses.values[:, 0, :]

def call_loss(
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        outputs: torch.Tensor,
        targets: torch.Tensor,
        cfg
        ) -> torch.Tensor:
    # we don't use the low_res_masks output, so we ignore it in the below line
    masks, iou_predictions, _ = outputs
    # we need to match the target dimensions to the mask dimensions, by repeating the target:
    targets = targets.repeat(1, masks.shape[1], 1, 1)
    masks_losses = loss_function(masks, targets)
    # iou loss is calculated using MSE loss, just like done in https://arxiv.org/pdf/2304.02643.pdf
    # to detach or not to detach, that is the question.
    # I think detaching makes sense, because we don't want to optimize the mask to fit the iou prediction.
    # We want to optimize the iou prediction to fit the mask.
    iou_targets = iou(masks, targets).detach()
    iou_losses = mse(iou_predictions, iou_targets)
    zipped_losses = torch.cat([masks_losses.unsqueeze(-1), iou_losses.unsqueeze(-1)], dim=2)
    # and now, take the minimum loss for each item in the batch:
    # zipped_losses has shape [batch_size, num_masks, 2]. We are looking for the tuple with the smallest first element.
    zipped_minimum_losses = find_minimum_loss(zipped_losses)
    # we end up with shape [batch_size, 2]. now sum each tuple:
    losses = torch.vmap(torch.sum)(zipped_minimum_losses)
    # return losses
    return reduce_losses(losses, cfg)

def reduce_losses(loss, cfg):
    reduction = cfg.model.loss.reduction
    if reduction == 'mean':
        return torch.mean(loss, dim=0)
    if reduction == 'sum':
        return torch.sum(loss, dim=0)

