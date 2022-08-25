from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


class RandomSplitter:
    def __init__(self):
        pass

    def split(
        self,
        dataset: pd.DataFrame,
        train_size: float = 0.8,
        test_size: float = 0.15,
        val_size: float = 0.05,
        seed: int = None,
    ):
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
        dataset.loc[train_ids, "step"] = "train"
        dataset.loc[val_ids, "step"] = "val"
        dataset.loc[test_ids, "step"] = "test"

        return dataset


class ScaffoldSplitter:
    def __init__(self) -> None:
        pass

    def _generate_scaffold(self, smiles: str, include_chirality: bool = False) -> str:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold

    def generate_scaffolds(self, dataset: pd.DataFrame, smiles_column: str = "smiles"):
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
            for (scaffold, scaffold_set) in sorted(
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
        seed: int = None,
    ):
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
        dataset.loc[train_ids, "step"] = "train"
        dataset.loc[val_ids, "step"] = "val"
        dataset.loc[test_ids, "step"] = "test"

        return dataset
