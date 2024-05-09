import torch
from finetune.train import get_point_prompts, get_random_point_prompts


def assert_close(a, b):
    torch.testing.assert_close(torch.tensor(a), torch.tensor(b))


def test_point_prompts():
    mask = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]]]])
    point_prompts, point_prompts_labels = get_point_prompts(mask)
    assert_close(point_prompts, [[[1,1]]])
    assert_close(point_prompts_labels, [[0]])
    

    mask = torch.tensor([[[[0,0,0],[0,1,0],[0,0,0]]]])
    point_prompts, point_prompts_labels = get_point_prompts(mask)
    assert_close(point_prompts, [[[1,1]]])
    assert_close(point_prompts_labels, [[1]])


    mask = torch.tensor([[[[0,1,0],[1,1,1],[0,1,0]]]])
    point_prompts, point_prompts_labels = get_point_prompts(mask)
    assert_close(point_prompts, [[[1,1]]])
    assert_close(point_prompts_labels, [[1]])

    mask = torch.tensor([[[[1,0,0],[1,1,0],[1,0,0]]]])
    point_prompts, point_prompts_labels = get_point_prompts(mask)
    assert_close(point_prompts, [[[1,0]]])
    assert_close(point_prompts_labels, [[1]])

    mask = torch.tensor([[[[1,1,1],[0,1,0],[0,0,0]]]])
    point_prompts, point_prompts_labels = get_point_prompts(mask)
    assert_close(point_prompts, [[[0,1]]])
    assert_close(point_prompts_labels, [[1]])



def test_random_point_prompts():
    mask = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]]]])
    point_prompts, point_prompts_labels = get_random_point_prompts(mask)
    y,x = point_prompts[0][0].to(int)
    assert mask[0][0][y][x] == 0

    mask = torch.tensor([[[[0,0,0],[0,1,0],[0,0,0]]]])
    point_prompts, point_prompts_labels = get_random_point_prompts(mask)
    assert_close(point_prompts, [[[1,1]]])
    assert_close(point_prompts_labels, [[1]])

    mask = torch.tensor([[[[0,1,0],[1,1,1],[0,1,0]]]])
    point_prompts, point_prompts_labels = get_random_point_prompts(mask)
    y,x = point_prompts[0][0].to(int)
    assert mask[0][0][y][x]
    assert_close(point_prompts_labels, [[1]])

    mask = torch.tensor([[[[1, 0, 0], [1, 1, 0], [1, 0, 0]]]])
    point_prompts, point_prompts_labels = get_random_point_prompts(mask)
    y,x = point_prompts[0][0].to(int)
    assert mask[0][0][y][x]
    assert_close(point_prompts_labels, [[1]])

    mask = torch.tensor([[[[1,1,1],[0,1,0],[0,0,0]]]])
    point_prompts, point_prompts_labels = get_random_point_prompts(mask)
    y,x = point_prompts[0][0].to(int)
    assert mask[0][0][y][x]
    assert_close(point_prompts_labels, [[1]])
