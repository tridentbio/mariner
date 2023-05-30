"""
Tests fleet.utils.data submodule.
"""
import pandas as pd

from fleet.dataset_schemas import DatasetConfig
from fleet.utils.data import build_columns_numpy


def test_build_columns_numpy2():
    """
    Tests build_columns_numpy function.
    """
    yaml_str = """
name: Small Zinc dataset
targetColumns:
  - name: tpsa
    dataType:
      domainKind: numeric
    outModule: LinearJoined
featureColumns:
  - name: smiles
    dataType:
      domainKind: smiles
  - name: mwt
    dataType:
      domainKind: numeric
featurizers:
  - name: MolToGraphFeaturizer
    type: fleet.model_builder.featurizers.MoleculeFeaturizer
    forwardArgs:
      mol: $smiles
    constructorArgs:
      allow_unknown: false
      sym_bond_list: true
      per_atom_fragmentation: false
"""
    config = DatasetConfig.from_yaml_str(yaml_str)
    dataset_path = "tests/data/csv/zinc_extra.csv"
    df = pd.read_csv(dataset_path)
    df = build_columns_numpy(config, df)
    assert "MolToGraphFeaturizer" in df.columns, "Missing featurized column in df"
