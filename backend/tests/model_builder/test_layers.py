import torch

from model_builder.layers import OneHot


def test_OneHot_int():
    x1 = [
        1,
        3,
        2,
    ]
    one_hot = OneHot()
    one_hot.classes = {1: 0, 2: 1, 3: 2}
    result = one_hot(x1)
    assert torch.eq(result, torch.Tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]])).sum()
