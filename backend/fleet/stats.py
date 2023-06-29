"""
Histograms generation and dataset statistics
"""
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from Bio.SeqUtils import molecular_weight as calc_molecular_weight
from pandas.core.frame import DataFrame
from rdkit.Chem import Descriptors

from fleet.model_builder.constants import TrainingStep
from mariner.schemas.dataset_schemas import ColumnsDescription, StatsType


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
    has_chiral_centers = (
        True if len(Chem.FindMolChiralCenters(mol)) > 0 else False
    )

    return mol_weight, mol_tpsa, atom_count, ring_count, has_chiral_centers


def get_biological_props(
    row: str,
    metadata: ColumnsDescription,
) -> Tuple[int, Optional[float], int, Optional[float]]:
    """Gets statistics of a biological input from.

    Data returned:

    Args:
        row: DNA, RNA or protein sequence string.
        metadata: column description the row must belong to.

    Returns:
        Tuple (sequence_length, gc_content, gaps_number, mwt)
        1. sequence_length: The length of the sequence.
        2. gc_content: The number of GC pairs
        3. gaps_number: The number of gaps in the sequence.
        4. mwt: The molecular weight of the sequence.
    """
    sequence_lengh, gc_content, gaps_number, mwt = None, None, None, None

    sequence_lengh = len(re.sub(r"[-\*]", "", row))
    gaps_number = len(re.findall(r"-", row))

    if (
        metadata.data_type.domain_kind in ["dna", "rna"]
        and not metadata.data_type.is_ambiguous
    ):
        gc_content = len(re.findall(r"[GC]", row)) / sequence_lengh
        mwt = calc_molecular_weight(
            row, seq_type=metadata.data_type.domain_kind.upper()
        )

    return sequence_lengh, gc_content, gaps_number, mwt


def create_categorical_histogram(
    data: pd.Series,
) -> Dict[Literal["values"], List[dict]]:
    """Creates the data used to generate a histogram from a pd.Series
    (dataset column).

    This method should only be used for columns with categorical data.

    Args:
        data (pd.Series): Dataset column used to generate the histogram.

    Returns:
        List[Dict[str, Any]]: List of dictionaries used to generate the
            histogram in vega-lite format.
    """

    # Calculate the histogram bins
    histogram = data.value_counts()

    histogram_data = []

    for index, count in enumerate(histogram):
        data_point = {
            "label": str(histogram.index[index]),
            "count": int(count),
        }

        histogram_data.append(data_point)

    return {"values": histogram_data[:15]}


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
            data,
            bins=np.linspace(
                data.min() - 0.5, data.max() + 0.5, data_range + 2
            ),
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
    dataset: pd.DataFrame,
    smiles_columns: List[str] = [],
    biological_columns: List[Dict[str, Any]] = [],
    categorical_columns: List[str] = [],
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
            statistics[column] = {
                "hist": create_float_histogram(dataset[column], 15)
            }
        elif column in categorical_columns:
            statistics[column] = {
                "hist": create_categorical_histogram(dataset[column])
            }
        elif np.issubdtype(dtype, int):
            statistics[column] = {
                "hist": create_int_histogram(dataset[column], 15)
            }
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
        chem_dataset["has_chiral_centers"] = chem_dataset[
            "has_chiral_centers"
        ].apply(
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

    for biological_column in biological_columns:
        (sequence_lengh, gc_content, gaps_number, mwt) = zip(
            *dataset[biological_column["col"]].apply(
                lambda row: get_biological_props(
                    row, metadata=biological_column["metadata"]
                )
            )
        )
        bio_dataset = DataFrame(
            {
                "sequence_lengh": sequence_lengh,
                "gc_content": gc_content,
                "gaps_number": gaps_number
                if not all(row == 0 for row in gaps_number)
                else None,
                "mwt": mwt,
            }
        )

        for column, dtype in bio_dataset.dtypes.items():
            if np.issubdtype(dtype, float):
                statistics[biological_column["col"]][column] = {
                    "hist": create_float_histogram(bio_dataset[column], 15)
                }
            elif np.issubdtype(dtype, int):
                statistics[biological_column["col"]][column] = {
                    "hist": create_int_histogram(bio_dataset[column], 15)
                }

    return statistics


def get_stats(
    dataset: pd.DataFrame,
    smiles_columns: List[str],
    biological_columns: List[Dict[str, Any]],
    categorical_columns: List[str],
) -> StatsType:
    """Computes the dataset histograms

    Args:
        dataset (pd.DataFrame): _description_
        smiles_columns (List[str]): _description_

    Returns:
        Stats:
            Histogram of the columns where it is calculable.
            It is separated by the step of the dataset, possible
            keys are: full, train, test, val
    """
    stats: StatsType = {}

    stats["full"] = get_dataset_summary(
        dataset, smiles_columns, biological_columns, categorical_columns
    )
    stats["train"] = get_dataset_summary(
        dataset[dataset["step"] == TrainingStep.TRAIN.value],
        smiles_columns,
        biological_columns,
        categorical_columns,
    )
    stats["test"] = get_dataset_summary(
        dataset[dataset["step"] == TrainingStep.TEST.value],
        smiles_columns,
        biological_columns,
        categorical_columns,
    )
    stats["val"] = get_dataset_summary(
        dataset[dataset["step"] == TrainingStep.VAL.value],
        smiles_columns,
        biological_columns,
        categorical_columns,
    )

    return stats


def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts metadata from the dataset to be cached on
    the entity

    Args:
        df (pd.DataFrame): Dataset to extract metadata from

    Returns:
        pd.DataFrame: Metadata dataframe
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
