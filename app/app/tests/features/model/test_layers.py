import torch

from app.features.model.layers import Concat
from app.features.model.layers.one_hot import OneHot


def test_Concat():
    x1 = torch.Tensor([1.0, 2.0, 3.0])
    x2 = torch.Tensor([1.0, 2.0, 3.0])
    concat_layer = Concat()
    y = concat_layer(x1, x2)
    assert torch.equal(y, torch.Tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))


def test_Concat_other_case():
    x1 = torch.Tensor([[1.0, 2.0, 3.0]])
    x2 = torch.Tensor([[1.0, 2.0, 3.0, 4.0]])
    concat_layer = Concat()
    y = concat_layer(x1, x2)
    assert torch.equal(y, torch.Tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0]]))


def test_OneHot_int():
    x1 = [
            1,
            3,
            2,
        ]
    one_hot = OneHot()
    one_hot.classes = {
        1: 0,
        2: 1,
        3: 2
    }
    result = one_hot(x1)
    assert torch.eq(
        result, torch.Tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    ).sum()
