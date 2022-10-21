# flake8: noqa
from typing import Dict, List, Union

import numpy as np
import rdkit.Chem as Chem
import torch
from rdkit.Chem.rdchem import Atom as RDKitAtom
from rdkit.Chem.rdchem import Bond as RDKitBond
from rdkit.Chem.rdchem import Mol as RDKitMol
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data as PyGData


class MoleculeFeaturizer:
    """
    Small molecule featurizer.
    Args:
        allow_unknown (bool, optional): Boolean indicating whether to add an additional feature for out-of-vocabulary one-hot encoded
            features.
        sym_bond_list (bool, optional): Boolean indicating whether bond list should be stored symmetrically.
        per_atom_fragmentation (bool, optional): if `per_atom_fragmentation=true` then multiple fragments of the molecule
            will be generated, the atoms will be removed one at a time resulting in a batch of fragments.
    """

    NODE_FEATURE_SLICES: Dict[str, slice] = {}
    EDGE_FEATURE_SLICES: Dict[str, slice] = {}

    def __init__(
        self,
        allow_unknown: bool,
        sym_bond_list: bool,
        per_atom_fragmentation: bool,
    ):
        super().__init__()

        # Allow unknown featurizations?
        self.allow_unknown = allow_unknown

        # Make bond list symmetric?
        self.sym_bond_list = sym_bond_list

        # Perform a per atom fragmentation?
        self.per_atom_fragmentation = per_atom_fragmentation

        # Atom feature lists
        self.ATOM_TYPE_SET = [
            "B",
            "C",
            "N",
            "O",
            "F",
            "Si",
            "P",
            "S",
            "Cl",
            "Br",
            "Sn",
            "I",
        ]
        self.CHIRALITY_SET = ["R", "S", "N/A"]
        self.HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
        self.TOTAL_NUM_HS_SET = [0, 1, 2, 3, 4]

        # Bond feature lists
        self.BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOE", "STEREOZ"]
        self.BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

        # Default feature sets
        self.ATOM_FEATURE_SETS = {
            # Atom features
            "atom_type": True,
            "chirality": True,
            "hybridization": True,
            "total_num_hs": True,
            "formal_charge": True,
        }

        self.BOND_FEATURE_SETS = {
            # Bond features
            "bond_type": True,
            "bond_stereo": True,
            "bond_conjugated": True,
        }

        # Functions to get required feature sets
        self.feature_functions = {
            # Atom features
            "atom_type": self._get_atom_type_one_hot,
            "chirality": self._get_atom_chirality_one_hot,
            "hybridization": self._get_atom_hybridization_one_hot,
            "total_num_hs": self._get_atom_num_hs_one_hot,
            "formal_charge": self._get_atom_formal_charge,
            # Bond features
            "bond_type": self._get_bond_type_one_hot,
            "bond_stereo": self._get_bond_stereo_one_hot,
            "bond_conjugated": self._get_bond_is_conjugated,
        }

        self.get_slices()

    def __call__(self, mol: Union[RDKitMol, str]) -> PyGData:
        return self._featurize(mol, self.sym_bond_list)

    def __repr__(self):
        atom_features = "atom = {}".format(
            ", ".join([f"{i}: {s}" for i, s in self.NODE_FEATURE_SLICES.items()])
        )
        bond_features = "bond = {}".format(
            ", ".join([f"{i}: {s}" for i, s in self.EDGE_FEATURE_SLICES.items()])
        )
        return f"Featurizer({atom_features}; {bond_features})"

    def get_slices(self):
        if self.allow_unknown:
            unknown_val = 1
        else:
            unknown_val = 0

        # Map feature sets
        atom_feature_vectors = {
            "atom_type": len(self.ATOM_TYPE_SET) + unknown_val,
            "chirality": len(self.CHIRALITY_SET) + unknown_val,
            "hybridization": len(self.HYBRIDIZATION_SET) + unknown_val,
            "total_num_hs": len(self.TOTAL_NUM_HS_SET) + unknown_val,
            "formal_charge": 1,
        }

        start = 0
        for feature_set in atom_feature_vectors:
            if self.ATOM_FEATURE_SETS[feature_set]:
                end = start + atom_feature_vectors[feature_set]
                self.NODE_FEATURE_SLICES[feature_set] = slice(start, end)
                start = end

        bond_feature_vectors = {
            "bond_type": len(self.BOND_TYPE_SET) + unknown_val,
            "bond_stereo": len(self.BOND_STEREO_SET) + unknown_val,
            "bond_conjugated": 1,
        }

        start = 0
        for feature_set in bond_feature_vectors:
            if self.BOND_FEATURE_SETS[feature_set]:
                end = start + bond_feature_vectors[feature_set]
                self.EDGE_FEATURE_SLICES[feature_set] = slice(start, end)
                start = end

    @staticmethod
    def _one_hot_encode(
        val: Union[int, str],
        allowable_set: Union[List[str], List[int]],
        include_unknown_set: bool = False,
    ) -> List[float]:

        # Init an one-hot vector
        if include_unknown_set is False:
            one_hot_length = len(allowable_set)
        else:
            one_hot_length = len(allowable_set) + 1

        one_hot = [0.0 for _ in range(one_hot_length)]

        # Try setting correct index to 1.0
        try:
            one_hot[allowable_set.index(val)] = 1.0
        except ValueError:
            if include_unknown_set:

                # If include_unknown_set is True, set the last index is 1.
                one_hot[-1] = 1.0
            else:
                pass

        return one_hot

    #################
    # Atom Features #
    #################

    def _get_atom_type_one_hot(self, atom: RDKitAtom) -> List[float]:
        return self._one_hot_encode(
            str(atom.GetSymbol()), self.ATOM_TYPE_SET, self.allow_unknown
        )

    def _get_atom_hybridization_one_hot(self, atom: RDKitAtom) -> List[float]:
        return self._one_hot_encode(
            str(atom.GetHybridization()), self.HYBRIDIZATION_SET, self.allow_unknown
        )

    @staticmethod
    def _get_atom_formal_charge(atom: RDKitAtom) -> List[float]:
        return [float(atom.GetFormalCharge())]

    def _get_atom_chirality_one_hot(self, atom: RDKitAtom) -> List[float]:
        try:
            configuration = atom.GetProp("_CIPCode")
        except KeyError:
            configuration = "N/A"

        return self._one_hot_encode(
            str(configuration), self.CHIRALITY_SET, self.allow_unknown
        )

    def _get_atom_num_hs_one_hot(self, atom: RDKitAtom) -> List[float]:
        return self._one_hot_encode(
            int(atom.GetTotalNumHs()), self.TOTAL_NUM_HS_SET, self.allow_unknown
        )

    def _get_mol_atom_features(self, mol: RDKitMol) -> List[float]:
        """Receive a RDKitMol object and returns a List of all atom features.
        Args:
            mol (RDKitMol): object that have the atoms featurized
        Returns:
            List[float]: list with the mol atom features
        """
        mol_atom_features = []
        # features
        included_atom_features = [
            feature_set
            for feature_set, include in self.ATOM_FEATURE_SETS.items()
            if include
        ]
        # featurize all atoms
        for atom in mol.GetAtoms():
            atom_features = []

            for feature_set in included_atom_features:
                features = self.feature_functions[feature_set](atom)
                atom_features.extend(features)

            mol_atom_features.append(atom_features)
        return mol_atom_features

    #################
    # Bond Features #
    #################

    def _get_bond_type_one_hot(self, bond: RDKitBond) -> List[float]:
        return self._one_hot_encode(
            str(bond.GetBondType()), self.BOND_TYPE_SET, self.allow_unknown
        )

    def _get_bond_stereo_one_hot(self, bond: RDKitBond) -> List[float]:
        return self._one_hot_encode(
            str(bond.GetStereo()), self.BOND_STEREO_SET, self.allow_unknown
        )

    @staticmethod
    def _get_bond_is_conjugated(bond: RDKitBond) -> List[float]:
        return [int(bond.GetIsConjugated())]

    @staticmethod
    def _get_bond_indices(bond: RDKitBond, symmetric: bool = False) -> List[List[int]]:
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        if symmetric:
            bond_list = [[start, end], [end, start]]
        else:
            bond_list = [[start], [end]]

        return bond_list

    #############
    # Featurize #
    #############

    def _featurize_frag(self, mol: RDKitMol, sym_bond_list: bool) -> PyGBatch:
        """Removes one atom at a time, generating a Batch object with all fragments.
        Args:
            mol (rdkit.Chem.rdchem.Mol): Mol object that will be fragmented.
            sym_bond_list (bool): Boolean indicating whether bond list should be stored symmetrically.
        Returns:
            torch_geometric.data.Batch: Batch object with set of Data
                objects representing the fragments.
        Examples:
            >>> from graphene.dataloader.featurizers import MoleculeFeaturizer
            >>> featurizer = MoleculeFeaturizer(per_atom_fragmentation=True)
            >>> featurizer('N[C@@H](C)C(=O)O')
        """
        # used for create PyGBatch object
        frags = []

        # All atoms features are needed to enumerate and remove one at a time
        mol_atom_features = self._get_mol_atom_features(mol)

        # Bond features to include
        included_bond_features = [
            feature_set
            for feature_set, include in self.BOND_FEATURE_SETS.items()
            if include
        ]

        for i, _ in enumerate(mol_atom_features):
            # For each atom, generate a frag without it
            new_mol_atom_features = np.delete(mol_atom_features, (i), axis=0)
            # Generate only the mol bond features for this fragment
            mol_bond_features = []
            mol_bond_list = [[], []]

            # Bond features still have to be created here because there are difference in conditions
            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()

                if start != i and end != i:
                    bond_features = []

                    for feature_set in included_bond_features:
                        features = self.feature_functions[feature_set](bond)
                        bond_features.extend(features)

                    mol_bond_features.append(bond_features)

                    # Handle symmetric bond information
                    if sym_bond_list:
                        mol_bond_features.append(bond_features)

                    starts, ends = self._get_bond_indices(bond, sym_bond_list)

                    # reindexing the bonds
                    if sym_bond_list:
                        for j, pos in enumerate(starts):
                            if pos > i:
                                starts[j] = pos - 1
                        for j, pos in enumerate(ends):
                            if pos > i:
                                ends[j] = pos - 1
                    else:
                        if starts[0] > i:
                            starts[0] -= 1
                        if ends[0] > i:
                            ends[0] -= 1

                    mol_bond_list[0].extend(starts)
                    mol_bond_list[1].extend(ends)

            out_dict = {
                "atom_features": torch.as_tensor(
                    np.array(new_mol_atom_features, dtype=np.float32)
                ),
                "bond_features": torch.as_tensor(
                    np.array(mol_bond_features, dtype=np.float32)
                ),
                "bond_list": torch.as_tensor(np.array(mol_bond_list, dtype=np.int64)),
            }

            frags.append(
                PyGData(
                    x=out_dict["atom_features"],
                    edge_index=out_dict["bond_list"],
                    edge_attr=out_dict["bond_features"],
                )
            )

        return PyGBatch.from_data_list(frags)

    def _featurize_mol(self, mol: RDKitMol, sym_bond_list: bool) -> PyGData:
        """Default method for featurize a molecule.
        Args:
            mol (rdkit.Chem.rdchem.Mol): Mol object that will be featurized.
            sym_bond_list (bool): Boolean indicating whether bond list should be stored symmetrically.
        Returns:
            torch_geometric.data.Data: Data object representing the molecule.
        Example:
        >>> from graphene.dataloader.featurizers import MoleculeFeaturizer
        >>> featurizer = MoleculeFeaturizer()
        >>> featurizer('N[C@@H](C)C(=O)O')
        """
        # Initialize lists to hold data
        mol_atom_features = self._get_mol_atom_features(mol)
        mol_bond_features = []
        mol_bond_list = [[], []]

        # Featurize all bonds
        included_bond_features = [
            feature_set
            for feature_set, include in self.BOND_FEATURE_SETS.items()
            if include
        ]

        for bond in mol.GetBonds():
            bond_features = []

            for feature_set in included_bond_features:
                features = self.feature_functions[feature_set](bond)
                bond_features.extend(features)

            mol_bond_features.append(bond_features)

            # Handle symmetric bond information
            if sym_bond_list:
                mol_bond_features.append(bond_features)

            starts, ends = self._get_bond_indices(bond, sym_bond_list)
            mol_bond_list[0].extend(starts)
            mol_bond_list[1].extend(ends)

        # Create the PyGData object
        out_dict = {
            "atom_features": torch.as_tensor(
                np.array(mol_atom_features, dtype=np.float32)
            ),
            "bond_features": torch.as_tensor(
                np.array(mol_bond_features, dtype=np.float32)
            ),
            "bond_list": torch.as_tensor(np.array(mol_bond_list, dtype=np.int64)),
        }

        return PyGData(
            x=out_dict["atom_features"],
            edge_index=out_dict["bond_list"],
            edge_attr=out_dict["bond_features"],
        )

    def _featurize(
        self, mol: Union[RDKitMol, str], sym_bond_list: bool = False
    ) -> Union[PyGData, PyGBatch]:
        # Read SMILEs string if necessary
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        if self.per_atom_fragmentation:
            # return and break exec
            return self._featurize_frag(mol, sym_bond_list)

        return self._featurize_mol(mol, sym_bond_list)
