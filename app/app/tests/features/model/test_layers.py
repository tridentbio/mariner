import torch

from app.features.model.layers import Concat


def test_Concat():
    x1 = torch.Tensor([1.0, 2.0, 3.0])
    x2 = torch.Tensor([1.0, 2.0, 3.0])
    concat_layer = Concat()
    y = concat_layer(x1, x2)
    assert torch.equal(y, torch.Tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))
