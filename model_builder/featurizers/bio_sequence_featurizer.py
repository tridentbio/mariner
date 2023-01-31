"""
Featurizers for biological data types
"""
import torch

from model_builder.featurizers.base_featurizers import ReversibleFeaturizer


class SequenceFeaturizer(ReversibleFeaturizer[str]):
    """Sequnce base featurizer. All biological sequence
    featurizer inherit from this class by specifying
    it's own alphabet dictionary
    """

    alphabet: dict[str, int]

    def __call__(self, input_: str) -> torch.Tensor:

        return torch.tensor([self.alphabet[i] for i in input_], dtype=torch.long)

    def undo(self, input_: torch.Tensor) -> str:
        raise NotImplementedError()


class DNAFeaturizer(SequenceFeaturizer):
    """DNA featurizer"""

    alphabet = {"A": 1, "C": 2, "G": 3, "T": 4}


class RNAFeaturizer(SequenceFeaturizer):
    """RNA featurizer"""

    alphabet = {"A": 1, "C": 2, "G": 3, "T": 4}


class ProteinFeaturizer(SequenceFeaturizer):
    """Protein sequence featurizers"""

    alphabet = {}
