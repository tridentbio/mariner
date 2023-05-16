import pytest

from fleet.model_builder.utils import (
    DataInstance,
    get_class_from_path_string,
    get_ref_from_data_instance,
)


class Baz:
    def __init__(self, b: int):
        self.b = b


class Foo:
    def __init__(self, a: Baz):
        self.a = a


def test_get_ref_from_data_instace():
    input = DataInstance()
    element = Foo(a=Baz(b=4))

    input["x"] = element

    assert get_ref_from_data_instance("x", input) == element
    assert get_ref_from_data_instance("x.a", input) == element.a
    assert get_ref_from_data_instance("x.a.b", input) == element.a.b


@pytest.mark.parametrize(
    "class_path",
    [
        "torch.nn.Linear",
        "torch_geometric.nn.GCNConv",
        "fleet.model_builder.layers.OneHot",
    ],
)
def test_get_class_from_path_string(class_path: str):
    try:
        get_class_from_path_string(class_path)
    except Exception as exc:
        pytest.fail("Failed with %r" % exc)
