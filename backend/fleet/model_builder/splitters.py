"""
Dataset splitting strategies
"""
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from .constants import TrainingStep


def apply_split_indexes(
    df: pd.DataFrame,
    split_type: Literal["random", "scaffold"],
    split_target: str,
    split_column: Union[str, None] = None,
):
    """Separates dataframe row into training, testing and validation

    Adds a 'step' column to the current dataframe, associating the row to
    some data science model stage, i.e. training, testing or validation

    Args:
        split_type: "random" or "scaffold", strategy used to to split examples
        split_target: The wanted distribution of training/validation/testing
        split_column: Only required for scaffold, must be a columns of Smiles
        data type

    Raises:
        NotImplementedError: in case the split_type is not recognized
    """
    train_size, val_size, test_size = map(
        lambda x: int(x) / 100, split_target.split("-")
    )
    if split_type == "random":
        splitter = RandomSplitter()
        df = splitter.split(df, train_size, test_size, val_size)
    elif split_type == "scaffold":
        splitter = ScaffoldSplitter()
        assert (
            split_column is not None
        ), "split column can't be none when split_type is scaffold"
        df = splitter.split(df, split_column, train_size, test_size, val_size)
    else:
        raise NotImplementedError(f"{split_type} splitting is not implemented")


class RandomSplitter:
    """
    Randomly splits dataset into train-validation-test
    """

    def __init__(self):
        pass

    def split(
        self,
        dataset: pd.DataFrame,
        train_size: float = 0.8,
        test_size: float = 0.15,
        val_size: float = 0.05,
        seed: Optional[int] = None,
    ):
        """Splits dataset randomly

        Args:
            dataset: DataFrame to be splitted
            train_size: train percents
            test_size: test percents
            val_size: validation percents
            seed: controm randomness
        """
        np.testing.assert_almost_equal(train_size + val_size + test_size, 1.0)

        if seed is not None:
            np.random.seed(seed)

        num_datapoints = len(dataset)

        train_cut = int(train_size * num_datapoints)
        val_cut = int((train_size + val_size) * num_datapoints)

        # List of indexes for each row shuffled
        shuffled = np.random.permutation(range(num_datapoints))

        train_ids = shuffled[:train_cut]
        val_ids = shuffled[train_cut:val_cut]
        test_ids = shuffled[val_cut:]

        # Create an empty column to be used as a marker
        dataset["step"] = None
        # Fill the step column with each equivalent step
        dataset.loc[train_ids, "step"] = TrainingStep.TRAIN.value
        dataset.loc[val_ids, "step"] = TrainingStep.VAL.value
        dataset.loc[test_ids, "step"] = TrainingStep.TEST.value

        return dataset


class ScaffoldSplitter:
    """
    Splits dataset into train-validation-test trying to achieve same
    distributions based on molecular properties of dataset SMILES column
    """

    def __init__(self) -> None:
        pass

    def _generate_scaffold(self, smiles: str, include_chirality: bool = False) -> str:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold

    def generate_scaffolds(self, dataset: pd.DataFrame, smiles_column: str = "smiles"):
        """Create scaffolds based on ``smiles_column`` column of ``dataset``

        Args:
            dataset: DataFrame
            smiles_column: column with smiles
        """
        scaffolds: Dict[str, List[int]] = {}

        for i, smiles in enumerate(dataset[smiles_column]):
            scaffold = self._generate_scaffold(smiles=smiles)

            if scaffold not in scaffolds:
                scaffolds[scaffold] = [i]
            else:
                scaffolds[scaffold].append(i)

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffolds_sets = [
            scaffold_set
            for (_scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        return scaffolds_sets

    def split(
        self,
        dataset: pd.DataFrame,
        smiles_column: str = "smiles",
        train_size: float = 0.8,
        test_size: float = 0.1,
        val_size: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Splits dataset using scaffold strategy

        Args:
            dataset: DataFrame to be splitted
            smiles_column: str column to get molecule of the sample
            train_size: train percents
            test_size: test percents
            val_size: validation percents
            seed: controm randomness
        """
        np.testing.assert_almost_equal(train_size + test_size + val_size, 1.0)
        scaffold_sets = self.generate_scaffolds(dataset, smiles_column)

        if seed is not None:
            np.random.seed(seed)

        train_cut = train_size * len(dataset)
        val_cut = (train_size + val_size) * len(dataset)

        train_ids = []
        val_ids = []
        test_ids = []

        for scaffold_set in scaffold_sets:
            if len(train_ids) + len(scaffold_set) > train_cut:
                if len(train_ids) + len(val_ids) + len(scaffold_set) > val_cut:
                    test_ids += scaffold_set
                else:
                    val_ids += scaffold_set
            else:
                train_ids += scaffold_set

        # Create an empty column to be used as a marker
        dataset["step"] = None
        # Fill the step column with each equivalent step
        dataset.loc[train_ids, "step"] = TrainingStep.TRAIN.value
        dataset.loc[val_ids, "step"] = TrainingStep.VAL.value
        dataset.loc[test_ids, "step"] = TrainingStep.TEST.value

        return dataset
