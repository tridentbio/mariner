from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from pandas.core.frame import DataFrame
from rdkit.Chem import Descriptors

from model_builder.constants import TrainingStep


def get_chemical_props(smiles: str) -> tuple:
    """Computes all desired chemical properties of a specific column of the
    dataset.

    Args:
        dataset (pd.DataFrame): Dataset that contains the smiles column.
        smiles_column (str): Name of the column that contains the smiles used
            to compute the properties.

    Returns:
        pd.DataFrame: A new dataset with the properties computed for that
            specific column.
    """
    mol = Chem.MolFromSmiles(smiles)

    # Calculate each prop
    atom_count = mol.GetNumAtoms()
    mol_weight = Descriptors.ExactMolWt(mol)
    mol_tpsa = Descriptors.TPSA(mol)
    ring_count = mol.GetRingInfo().NumRings()
    has_chiral_centers = True if len(Chem.FindMolChiralCenters(mol)) > 0 else False

    return mol_weight, mol_tpsa, atom_count, ring_count, has_chiral_centers


def create_float_histogram(data: pd.Series, bins: int = 15) -> Dict[str, Any]:
    """Creates the data used to generate a histogram from a pd.Series
    (dataset column).

    This method should only be used for columns with continuous numeric data.

    Args:
        data (pd.Series): Dataset column used to generate the histogram.
        bins (int, optional): Number of bins used to generate the histogram.
            Defaults to 15.

    Returns:
        Dict[str, Any]: Dictionary used to generate the histogram in
            vega-lite format.
    """

    # Calculate the histogram bins
    histogram = np.histogram(data, bins=bins)

    histogram_data = {"values": []}

    for index, count in enumerate(histogram[0]):
        data_point = {
            "bin_start": histogram[1][index],
            "bin_end": histogram[1][index + 1],
            "count": int(count),
        }

        histogram_data["values"].append(data_point)

    histogram_data["values"].append(
        {
            "min": data.min(),
            "max": data.max(),
            "mean": data.mean(),
            "median": data.median(),
        }
    )

    return histogram_data


def create_int_histogram(data: pd.Series, bins: int) -> Dict[str, Any]:
    """Creates the data used to generate a histogram from a pd.Series
    (dataset column).

    This method should only be used for columns with categorical data.

    Args:
        data (pd.Series): Dataset column used to generate the histogram.
        bins (int): Number of bins used to generate the histogram.

    Returns:
        Dict[str, Any]: Dictionary used to generate the histogram in
            vega-lite format.
    """
    data_range = data.max() - data.min()

    if data_range <= bins:
        histogram = np.histogram(
            data, bins=np.linspace(data.min() - 0.5, data.max() + 0.5, data_range + 2)
        )
    else:
        histogram = np.histogram(data, bins=bins)

    histogram_data = {"values": []}

    for index, count in enumerate(histogram[0]):
        data_point = {
            "bin_start": histogram[1][index],
            "bin_end": histogram[1][index + 1],
            "count": int(count),
        }

        histogram_data["values"].append(data_point)

    histogram_data["values"].append(
        {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "median": float(data.median()),
        }
    )

    return histogram_data


def get_dataset_summary(
    dataset: pd.DataFrame, smiles_columns: List[str] = []
) -> dict[str, Union[pd.Series, dict[str, pd.Series]]]:
    """Computes the dataset histograms

    Creates a dictionary to hold histograms of float/int dtype columns
    and histograms of mol properties computed by rdkit

    :param dataset pd.DataFrame: original dataset
    :param smiles_columns List[str]: Optional array of smiles column to
    compute mol props
    :return dict[str, Union[pd.Series, dict[str, pd.Series]]
    """

    statistics = {}

    for column, dtype in dataset.dtypes.items():
        if np.issubdtype(dtype, float):
            statistics[column] = {"hist": create_float_histogram(dataset[column], 15)}
        elif np.issubdtype(dtype, int):
            statistics[column] = {"hist": create_int_histogram(dataset[column], 15)}
        else:
            statistics[column] = {}  # The key is used to recover all columns

    for smiles_column in smiles_columns:
        # Get chemical stats for smile column
        (mwts, tpsas, atom_counts, ring_counts, has_chiral_centers_arr) = zip(
            *dataset[smiles_column].apply(get_chemical_props)
        )

        chem_dataset = DataFrame(
            {
                "mwt": mwts,
                "tpsa": tpsas,
                "atom_count": atom_counts,
                "ring_count": ring_counts,
                "has_chiral_centers": has_chiral_centers_arr,
            }
        )

        # Convert boolean column to 0 and 1
        chem_dataset["has_chiral_centers"] = chem_dataset["has_chiral_centers"].apply(
            lambda x: 1 if x else 0,
        )

        statistics[smiles_column] = {}
        for column, dtype in chem_dataset.dtypes.items():
            if np.issubdtype(dtype, float):
                statistics[smiles_column][column] = {
                    "hist": create_float_histogram(chem_dataset[column], 15)
                }
            elif np.issubdtype(dtype, int):
                statistics[smiles_column][column] = {
                    "hist": create_int_histogram(chem_dataset[column], 15)
                }

    return statistics


def get_stats(dataset: pd.DataFrame, smiles_columns: List[str]):
    stats = {}

    stats["full"] = get_dataset_summary(dataset, smiles_columns)
    stats["train"] = get_dataset_summary(
        dataset[dataset["step"] == TrainingStep.TRAIN.value], smiles_columns
    )
    stats["test"] = get_dataset_summary(
        dataset[dataset["step"] == TrainingStep.TEST.value], smiles_columns
    )
    stats["val"] = get_dataset_summary(
        dataset[dataset["step"] == TrainingStep.VAL.value], smiles_columns
    )

    return stats


def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts metadata from the dataset to be cached on
    the entity
    """
    dtypes: pd.Series = pd.Series(
        list(map(str, df.dtypes)), name="types", index=df.columns
    )
    stats = df.describe(include="all")
    nacount: pd.Series = df.isna().sum()
    nacount.name = "na_count"
    nacount.index = df.columns
    stats = stats.append(dtypes)
    stats = stats.append(nacount)
    stats = stats.fillna("NaN")
    return stats
