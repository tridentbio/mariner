import pytest
from pydantic import BaseModel, ValidationError

from app.features.dataset.schema import (
    CategoricalDataType,
    ColumnsDescription,
    NumericalDataType,
    SmileDataType,
    Split,
    StringDataType,
)


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


class TestColumnDescription:
    descriptions_fixture = [
        {
            "data_type": NumericalDataType(domain_kind="numerical"),
            "pattern": "tpsa",
            "description": "TPSA",
        },
        {
            "data_type": StringDataType(domain_kind="string"),
            "pattern": "tpsa",
            "description": "TPSA",
        },
        {
            "data_type": CategoricalDataType(
                domain_kind="categorical", classes={"a": 0, "b1": 1}
            ),
            "pattern": "tpsa",
            "description": "TPSA",
        },
    ]

    @pytest.mark.parametrize("args", descriptions_fixture)
    def test_builds_successfully(self, args):
        try:
            col_description = ColumnsDescription(**args)
            assert col_description
            assert col_description.data_type.domain_kind
        except ValidationError as exc:
            assert False, f"raised an exception {exc}"

    valid_jsons_fixture = [
        (
            """
            {
                "pattern": "exp",
                "dataType": {
                    "domainKind": "numerical"
                },
                "description": "experiment measurement"
            }
        """,
            NumericalDataType,
            "numerical",
        ),
        (
            """
            {
                "pattern": "exp",
                "dataType": {
                    "domainKind": "smiles"
                },
                "description": "experiment measurement"
            }
        """,
            SmileDataType,
            "smiles",
        ),
        (
            """
            {
                "pattern": "exp",
                "dataType": {
                    "domainKind": "string"
                },
                "description": "experiment measurement"
            }
        """,
            StringDataType,
            "string",
        ),
        (
            """
            {
                "pattern": "exp",
                "dataType": {
                    "domainKind": "categorical",
                    "classes": {
                        "a": 0,
                        "b": 1
                    }
                },
                "description": "experiment measurement"
            }
        """,
            CategoricalDataType,
            "categorical",
        ),
    ]

    @pytest.mark.parametrize("json_expected", valid_jsons_fixture)
    def test_builds_from_json(self, json_expected):
        json, expected, domain_kind = json_expected
        try:
            col_description = ColumnsDescription.parse_raw(json)
            assert col_description.data_type.domain_kind == domain_kind
            assert isinstance(
                col_description.data_type, expected
            ), "Columns Description parsed with an invalid data type"
        except ValidationError:
            assert False, "ColumnDescription raised an exception parsing a valid json"

    invalid_jsons_fixture = [
        """
            {
                "pattern": "exp",
                "dataType": {
                    "domainKind": "juice"
                },
                "description": "experiment measurement"
            }
        """,
        """
            {
                "pattern": "exp",
                "dataType": {
                    "domainKind": "smile
                },
                "description": "experiment measurement"
            }
        """,
        """
            {
                "pattern": "exp",
                "dataType": {
                    "domainKind": "categorical"
                },
                "description": "experiment measurement"
            }
        """,
    ]

    @pytest.mark.parametrize("json", invalid_jsons_fixture)
    def test_raises_validation_error_on_invalid_jsons(self, json):
        with pytest.raises(ValidationError):
            ColumnsDescription.parse_raw(json)
