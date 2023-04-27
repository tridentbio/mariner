"""
Featurizers for biological data types
"""
import torch

from fleet.model_builder.featurizers.base_featurizers import ReversibleFeaturizer


class SequenceFeaturizer(ReversibleFeaturizer[str]):
    """Sequnce base featurizer. All biological sequence
    featurizer inherit from this class by specifying
    its own alphabet dictionary
    """

    alphabet: dict[str, int]
    inverse_alphabet: dict[int, str]

    def __init__(self):
        super(SequenceFeaturizer).__init__()

        # Invert the alphabet for decoding
        self.inverse_alphabet = {v: k for k, v in self.alphabet.items()}

        # Get the number of tokes in the alphabet for use with embedding layers later
        self.num_embeddings = len(self.alphabet.keys())

    def __call__(self, input_: str) -> torch.Tensor:
        """Featurize sequence"""
        return self.featurize(input_)

    def featurize(self, input_: str) -> torch.Tensor:
        """Featurize sequence to pytorch LongTensor"""
        # Initialize an empty sequence
        sequence = []

        # Loop through and featurize or raise ValueError if unrecognized token
        for i in input_:
            if i not in self.alphabet:
                raise ValueError(
                    f"Unrecognized token '{i}' for {self.__class__.__name__}."
                )

            sequence.append(self.alphabet[i])

        # Return the torch tensor
        return torch.tensor(sequence, dtype=torch.long)

    def unfeaturize(self, input_: torch.Tensor) -> str:
        """Unfeaturize pytorch LongTensor to sequence"""
        # Initialize an empty sequence
        sequence = []

        # Loop through and unfeaturize or raise ValueError if unrecognized token
        for i in input_:
            i = i.item()
            if i not in self.inverse_alphabet:
                raise ValueError(
                    f"Unrecognized value '{i}' for {self.__class__.__name__}."
                )

            sequence.append(self.inverse_alphabet[i])

        # Return the unfeaturized string
        return "".join(sequence)


class DNASequenceFeaturizer(SequenceFeaturizer):
    """DNA sequence featurizer"""

    alphabet = {
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4,
        "R": 5,
        "Y": 6,
        "S": 7,
        "W": 8,
        "K": 9,
        "M": 10,
        "B": 11,
        "D": 12,
        "H": 13,
        "V": 14,
        "N": 15,
        "-": 16,
    }


class RNASequenceFeaturizer(SequenceFeaturizer):
    """RNA sequence featurizer"""

    alphabet = {
        "A": 1,
        "C": 2,
        "G": 3,
        "U": 4,
        "R": 5,
        "Y": 6,
        "S": 7,
        "W": 8,
        "K": 9,
        "M": 10,
        "B": 11,
        "D": 12,
        "H": 13,
        "V": 14,
        "N": 15,
        "-": 16,
    }


class ProteinSequenceFeaturizer(SequenceFeaturizer):
    """Protein sequence featurizer"""

    alphabet = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "-": 21,
        "*": 22,
    }
