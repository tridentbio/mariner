from typing import List

import pandas as pd
import pytest
import pytest_asyncio

from mariner.core.config import settings
from mariner.ray_actors.dataset_transforms import DatasetTransforms
from mariner.schemas.dataset_schemas import ColumnsMeta

BIO_DATASET_PATH = "tests/data/csv/bio_rna_dna_protein.csv"
ZINC_DATASET_PATH = "tests/data/csv/zinc_extra.csv"
HIV_DATASET_PATH = "tests/data/csv/HIV.csv"
DATASETS_TO_TEST = [ZINC_DATASET_PATH, HIV_DATASET_PATH]


@pytest.mark.integration
class TestDatasetTransforms:
    @pytest.mark.parametrize(
        "csvpath",
        DATASETS_TO_TEST,
    )
    @pytest.mark.asyncio
    async def test_write_dataset_buffer(self, csvpath):
        dataset_ray_transformer = DatasetTransforms.remote()

        chunk_size = 1024
        try:
            with open(csvpath, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    await dataset_ray_transformer.write_dataset_buffer.remote(chunk)
                # dataset_ray_transformer.is_dataset_fully_loaded = True
                await dataset_ray_transformer.set_is_dataset_fully_loaded.remote(True)
                is_loaded = (
                    await dataset_ray_transformer.get_is_dataset_fully_loaded.remote()
                )
                assert is_loaded
                df = await dataset_ray_transformer.get_dataframe.remote()
                assert isinstance(df, pd.DataFrame)
        except Exception as exp:
            assert exp is None, "Failed to instantiate ray dataset transformer"

    async def make_transformer_fixture(self, csvpath: str):
        dataset_ray_transformer = DatasetTransforms.remote()
        with open(csvpath, "rb") as f:
            for chunk in iter(lambda: f.read(settings.APPLICATION_CHUNK_SIZE), b""):
                await dataset_ray_transformer.write_dataset_buffer.remote(chunk)
            await dataset_ray_transformer.set_is_dataset_fully_loaded.remote(True)
            return dataset_ray_transformer

    @pytest_asyncio.fixture
    async def bio_transformer_fixture(self):
        transformer = await self.make_transformer_fixture(BIO_DATASET_PATH)
        return transformer

    @pytest_asyncio.fixture
    async def hiv_transformer_fixture(self):
        transformer = await self.make_transformer_fixture(HIV_DATASET_PATH)
        return transformer

    @pytest_asyncio.fixture
    async def zinc_transformer_fixture(self):
        transformer = await self.make_transformer_fixture(ZINC_DATASET_PATH)
        return transformer

    @pytest.mark.asyncio
    async def test_get_columns_metadata(self, hiv_transformer_fixture):
        cols_metadata = await hiv_transformer_fixture.get_columns_metadata.remote()
        assert (
            len(cols_metadata) > 0
        ), "dataset transformer failed to get columns metadata"
        for col_meta in cols_metadata:
            assert isinstance(col_meta, ColumnsMeta)

    @pytest.mark.asyncio
    async def test_get_columns_metadata_hiv(self, hiv_transformer_fixture):
        (
            rows_count,
            cols_count,
            stats,
        ) = await hiv_transformer_fixture.get_entity_info_from_csv.remote()
        assert rows_count == 20
        assert cols_count == 4

        # check stats colulmns
        expected_stats_cols = ["smiles", "activity", "HIV_active", "step"]
        got_stats_cols = list(stats.keys())
        assert got_stats_cols == expected_stats_cols

        # check stats indexes
        expected_stats_indexes = [
            "count",
            "unique",
            "top",
            "freq",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "types",
            "na_count",
        ]
        got_stats_indexes = list(stats.index)
        assert got_stats_indexes == expected_stats_indexes

    @pytest.mark.asyncio
    async def test_get_columns_metadata_bio(self, bio_transformer_fixture):
        metadata_list: List[
            ColumnsMeta
        ] = await bio_transformer_fixture.get_columns_metadata.remote()
        assert len(metadata_list) == 5

        expected_domain = ["string", "rna", "dna", "protein", "categorical"]

        for i, metadata in enumerate(metadata_list):
            assert isinstance(metadata, ColumnsMeta)
            domain_kind = metadata.dtype.domain_kind
            assert (
                domain_kind == expected_domain[i]
            ), f"expected bio dataset column {i} domain kind to be {expected_domain[i]}"

            if domain_kind in ["dna", "rna"]:
                assert (
                    metadata.dtype.is_ambiguous
                ), f"expected bio dataset column {expected_domain[i]} to be ambiguous"

    @pytest.mark.asyncio
    async def test_apply_indexes_scaffold(self, hiv_transformer_fixture):
        initial_df = await hiv_transformer_fixture.get_dataframe.remote()
        # Remove step column from initial_df
        initial_df = initial_df.drop(columns=["step"])
        expected_len = len(initial_df.columns) + 1
        await hiv_transformer_fixture.apply_split_indexes.remote(
            split_type="scaffold", split_target="60-20-20", split_column="smiles"
        )
        got_df = await hiv_transformer_fixture.get_dataframe.remote()
        got_len = len(got_df.columns)
        assert (
            got_len == expected_len
        ), "Expected final dataframe to have one extra column"

    @pytest.mark.long
    @pytest.mark.asyncio
    async def test_get_dataset_summary(self, hiv_transformer_fixture):
        await hiv_transformer_fixture.apply_split_indexes.remote(
            split_type="random", split_target="60-20-20"
        )
        stats = await hiv_transformer_fixture.get_dataset_summary.remote()
        assert stats is not None
        expected_keys = ["full", "train", "test", "val"]
        for expected_key in expected_keys:
            assert (
                expected_key in stats
            ), f"Missing {expected_key} from dataset histograms"
