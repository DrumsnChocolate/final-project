from typing import Callable

import torch
import torchvision



def dice_single(output, target):
    eps = 1e-7
    return 2 * torch.sum(output * target) / (torch.sum(output * output) + torch.sum(target * target) + eps)


def dice_item(outputs, targets):
    return torch.vmap(dice_single)(outputs, targets)


def dice(output_batch, target_batch) -> torch.Tensor:
    assert output_batch.shape[1] in [1, 3], \
        f'Expected outputs to have 1 or 3 masks, but got {output_batch.shape[1]} masks'
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
    eps = 1e-7
    intersection = torch.sum(output * target > 0)
    union = torch.sum((output > 0).logical_or((target > 0)))
    return intersection / (union + eps)


def iou_item(outputs, targets):
    return torch.vmap(iou_single)(outputs, targets)


def iou(output_batch, target_batch) -> torch.Tensor:
    assert output_batch.shape[1] in [1, 3], \
        f'Expected outputs to have 1 or 3 masks, but got {output_batch.shape[1]} masks'
    assert target_batch.shape[1] == output_batch.shape[
        1], f'Expected targets to have {output_batch.shape[1]} masks, but got {target_batch.shape[1]} masks'
    return torch.vmap(iou_item)(output_batch, target_batch)


def mse(predicted_batch, target_batch):
    return torch.nn.functional.mse_loss(predicted_batch, target_batch, reduction='none')

def get_loss_function(loss_definition) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    loss_metric = None
    if loss_definition.name == 'IoU':
        loss_metric = lambda outputs, targets: iou(outputs, targets)
    if loss_definition.name == 'Dice':
        loss_metric = lambda outputs, targets: dice(outputs, targets)
    if loss_definition.name == 'Focal':
        alpha = loss_definition.get('alpha', -1)
        gamma = loss_definition.get('gamma', 2.0)
        reduction = loss_definition.get('reduction', 'mean')
        loss_metric = lambda outputs, targets: focal(outputs, targets.float(), alpha, gamma, reduction)
    if loss_metric is None:
        raise NotImplementedError()
    loss_function = loss_metric
    if loss_definition.get('invert', False):
        permitted_inversions = ['IoU', 'Dice']
        assert loss_definition.name in permitted_inversions, f'Invert is only supported for {permitted_inversions}, but got {loss_definition.name}'
        loss_function = lambda *args: 1 - loss_metric(*args)
    return loss_function


def build_loss_function(cfg):
    loss_parts = [get_loss_function(loss_item) for loss_item in cfg.model.loss.parts]
    loss_weights = [loss_item.weight for loss_item in cfg.model.loss.parts]
    total_weight = sum(loss_weights)
    return lambda outputs, targets: sum(
        [weight * loss_part(outputs, targets) for weight, loss_part in zip(loss_weights, loss_parts)]) / total_weight


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
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        cfg
) -> torch.Tensor:

    # we don't use the low_res_masks output, so we ignore it in the below line
    masks, iou_predictions, _ = outputs
    assert masks.isnan().sum() == 0, f'Predicted masks contain {masks.isnan().sum()} NaN values'
    assert iou_predictions.isnan().sum() == 0, f'Predicted iou values contain {iou_predictions.isnan().sum()} NaN values'
    # we need to match the target dimensions to the mask dimensions, by repeating the target:
    targets = targets.repeat(1, masks.shape[1], 1, 1)
    assert (targets > 0).sum(axis=(2, 3)).all()  # assert that all masks will have a target area > 0
    masks_losses = loss_function(masks, targets)
    # iou loss is calculated using MSE loss, just like done in https://arxiv.org/pdf/2304.02643.pdf
    # to detach or not to detach, that is the question.
    # I think detaching makes sense, because we don't want to optimize the mask to fit the model's own iou prediction;
    # We want to optimize the iou prediction to fit the mask.
    iou_targets = iou(masks, targets).detach()
    iou_losses = mse(iou_predictions, iou_targets)
    zipped_losses = torch.cat([masks_losses.unsqueeze(-1), iou_losses.unsqueeze(-1)], dim=2)
    # and now, take the minimum loss for each item in the batch:
    # zipped_losses has shape [batch_size, num_masks, 2]. We are looking for the tuple with the smallest first element.
    zipped_minimum_losses = find_minimum_loss(zipped_losses)
    # we end up with shape [batch_size, 2]. now sum each tuple:
    losses = torch.vmap(torch.sum)(zipped_minimum_losses)
    assert losses.isnan().sum() == 0, 'Losses contain NaN values'
    # return losses
    loss = reduce_losses(losses, cfg)
    assert loss.isnan().sum() == 0, 'Loss is NaN'
    return loss


def reduce_losses(loss, cfg):
    reduction = cfg.model.loss.reduction
    if reduction == 'mean':
        return torch.mean(loss, dim=0)
    if reduction == 'sum':
        return torch.sum(loss, dim=0)
