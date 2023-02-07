from model_builder.utils import DataInstance, get_ref_from_data_instance


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
