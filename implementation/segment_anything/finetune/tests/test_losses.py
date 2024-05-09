import torch

from ..loss import dice_single, iou_single, focal_single


def assert_close(a, b):
    torch.testing.assert_close(torch.tensor(a), torch.tensor(b))


def test_dice_single():
    eps = 1e-7

    output = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert dice_single(output, target) == 1  # perfect match, even though both are empty

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    assert dice_single(output, target) == 1  # perfect match


    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert_close(dice_single(output, target), eps / (eps + 1))

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    assert_close(dice_single(output, target), (2 + eps)/(10 + eps))


def test_iou_single():
    eps = 1e-7

    output = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert iou_single(output, target) == 1  # perfect match, even though both are empty

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    assert iou_single(output, target) == 1  # perfect match

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert_close(iou_single(output, target), eps / (1 + eps))

    output = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    target = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    assert_close(iou_single(output, target), (1 + eps) / (9 + eps))


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
