import torch
from ..metrics import dice_single, iou_single, focal_single


def test_dice_single():
    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    assert dice_single(output, target) == 1

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert dice_single(output, target) == 0

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    assert dice_single(output, target) == 2/10

    # this case SHOULD never occur in practice, but it's good to know what happens if it does.
    # the reason it never occurs in practice is that the target always contains at least one 1.
    output = torch.tensor([0])
    target = torch.tensor([0])
    assert torch.isnan(dice_single(output, target))

def test_iou_single():
    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    assert iou_single(output, target) == 1

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert iou_single(output, target) == 0

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    assert iou_single(output, target) == 1 / 9

    # again, this should never actually occur
    output = torch.tensor([0])
    target = torch.tensor([0])
    assert torch.isnan(iou_single(output, target))

def test_focal_single():
    alpha = -1
    gamma = 2.0
    reduction = 'mean'
    # expects floats
    output = torch.tensor([1.0])
    target = torch.tensor([1.0])
    loss_correct = focal_single(output, target, alpha, gamma, reduction)
    loss_certain_correct = focal_single(output*2, target, alpha, gamma, reduction)
    loss_undecisive = focal_single(torch.tensor([0.0]), target, alpha, gamma, reduction)
    loss_wrong = focal_single(-output, target, alpha, gamma, reduction)
    loss_certain_wrong = focal_single(-output*2, target, alpha, gamma, reduction)

    assert loss_correct > 0
    # assert that a more 'certain' correct prediction has a lower loss than a less certain correct prediction
    assert loss_certain_correct < loss_correct
    # assert that an undecisive prediction has a higher loss than a correct prediction
    assert loss_undecisive > loss_correct
    # assert that a wrong prediction has a higher loss than a correct prediction
    assert loss_wrong > loss_correct
    # assert that a wrong prediction has a higher loss than an undecisive prediction
    assert loss_wrong > loss_undecisive
    # assert that a more 'certain' wrong prediction has a higher loss than a less certain wrong prediction
    assert loss_certain_wrong > loss_wrong
    # assert that the difference between certain wrong and wrong is higher than between wrong and undecisive
    assert loss_certain_wrong - loss_wrong > loss_wrong - loss_undecisive
