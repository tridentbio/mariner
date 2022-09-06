from typing import Any, Dict
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors


def get_chemical_props(
    smiles: str
) -> tuple:
    """ Computes all desired chemical properties of a specific column of the
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
        True if len(Chem.FindMolChiralCenters(mol)) > 0
        else False
    )

    return mol_weight, mol_tpsa, atom_count, ring_count, has_chiral_centers


def create_float_histogram(data: pd.Series, bins: int = 15) -> Dict[str, Any]:
    """ Creates the data used to generate a histogram from a pd.Series
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

    histogram_data = {'values': []}

    for index, count in enumerate(histogram[0]):
        data_point = {
            'bin_start': histogram[1][index],
            'bin_end': histogram[1][index + 1],
            'count': int(count)
        }

        histogram_data['values'].append(data_point)

    histogram_data['values'].append({
        'min': data.min(),
        'max': data.max(),
        'mean': data.mean(),
        'median': data.median(),
    })

    return histogram_data


def create_int_histogram(data: pd.Series, bins: int) -> Dict[str, Any]:
    """ Creates the data used to generate a histogram from a pd.Series
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
                data.min() - 0.5,
                data.max() + 0.5,
                data_range + 2
            )
        )
    else:
        histogram = np.histogram(data, bins=bins)

    histogram_data = {'values': []}

    for index, count in enumerate(histogram[0]):
        data_point = {
            'bin_start': histogram[1][index],
            'bind_end': histogram[1][index + 1],
            'count': int(count)
        }

        histogram_data['values'].append(data_point)

    histogram_data['values'].append({
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'median': float(data.median()),
    })

    return histogram_data


def get_dataset_summary(dataset: pd.DataFrame, smiles_column: str):

    statistics = {}

    # Iterate over column and calculate the
    # distributions for each int or float column
    for column, dtype in dataset.dtypes.items():
        if np.issubdtype(dtype, float):
            statistics[column] = {
                'hist': create_float_histogram(dataset[column], 15)
            }
        elif np.issubdtype(dtype, int):
            statistics[column] = {
                'hist': create_int_histogram(dataset[column], 15)
            }

    # Calculate and compute chemical stats for smile column
    (
        dataset['mwt'], dataset['tpsa'], dataset['atom_count'],
        dataset['ring_count'], dataset['has_chiral_centers']
    ) = zip(*dataset[smiles_column].apply(get_chemical_props))

    chem_dataset = dataset[[
        'mwt', 'tpsa', 'atom_count', 'ring_count', 'has_chiral_centers'
    ]]

    # Convert boolean column to 0 and 1
    chem_dataset['has_chiral_centers'] = (
        chem_dataset['has_chiral_centers'].apply(lambda x: 1 if x else 0)
    )

    statistics[smiles_column] = {}
    for column, dtype in chem_dataset.dtypes.items():
        if np.issubdtype(dtype, float):
            statistics[smiles_column][column] = {
                'hist': create_float_histogram(chem_dataset[column], 15)
            }
        elif np.issubdtype(dtype, int):
            statistics[smiles_column][column] = {
                'hist': create_int_histogram(chem_dataset[column], 15)
            }

    return statistics


def get_stats(dataset: pd.DataFrame, smiles_column: str):
    stats = {}

    stats['full'] = get_dataset_summary(dataset, smiles_column)
    stats['train'] = get_dataset_summary(dataset[dataset['step'] == 'train'])
    stats['test'] = get_dataset_summary(dataset[dataset['step'] == 'test'])
    stats['val'] = get_dataset_summary(dataset[dataset['val'] == 'val'])

    return stats
