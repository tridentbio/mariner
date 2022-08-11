import pytest
from pydantic import BaseModel, ValidationError

from app.features.dataset.schema import Split


class TestSplit:
    def test_split_from_valid_string(self):
        class Foo(BaseModel):
            split: Split

        foo = Foo(split=Split("80-15-5"))
        assert foo.split.train_percents == 80
        assert foo.split.test_percents == 15
        assert foo.split.val_percents == 5

    def test_split_not_summing_100(self):
        with pytest.raises(ValidationError):

            class Foo(BaseModel):
                split: Split

            Foo(split=Split("81-15-5"))
