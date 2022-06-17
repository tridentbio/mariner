import torch

from app.features.model.layers import Concat

def test_Concat():
    x1 = torch.Tensor([1., 2., 3.])
    x2 = torch.Tensor([1., 2., 3.])
    concat_layer = Concat()
    y = concat_layer(x1, x2)
    assert torch.equal(y, torch.Tensor([1., 2., 3., 1., 2., 3.]))


