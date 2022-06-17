import torch

from app.features.model.layers.Concat import ConcatLayer

def test_Concat():
    x1 = torch.Tensor([1., 2., 3.])
    x2 = torch.Tensor([1., 2., 3.])
    concat_layer = ConcatLayer()
    y = concat_layer(x1, x2)
    assert torch.equal(y, torch.Tensor([1., 2., 3., 1., 2., 3.]))


