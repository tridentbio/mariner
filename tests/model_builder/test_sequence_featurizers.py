import pytest

import torch

from model_builder.featurizers import DNASequenceFeaturizer, RNASequenceFeaturizer, ProteinSequenceFeaturizer

@pytest.fixture
def dna_seq_feat():
    return DNASequenceFeaturizer()


@pytest.fixture
def rna_seq_feat():
    return RNASequenceFeaturizer()


@pytest.fixture
def protein_seq_feat():
    return ProteinSequenceFeaturizer()


def test_DNASequenceFeaturizer(dna_seq_feat):
    good_seq = "ACAGTRYSWKMBDHVN-"
    good_seq_expected = torch.tensor([1, 2, 1, 3, 4,
                                      5, 6, 7, 8, 9,
                                      10, 11, 12, 13,
                                      14, 15, 16],
                                      dtype=torch.long)

    

    assert torch.all(dna_seq_feat(good_seq) == good_seq_expected)
    assert dna_seq_feat.unfeaturize(good_seq_expected) == good_seq


def test_DNASequenceFeaturizer_bad_sequence(dna_seq_feat):
    bad_seq = "ACGTJ"

    with pytest.raises(ValueError, match=r"Unrecognized token 'J' for DNASequenceFeaturizer."):
        dna_seq_feat(bad_seq)


def test_RNASequenceFeaturizer(rna_seq_feat):
    good_seq = "ACAGURYSWKMBDHVN-"
    good_seq_expected = torch.tensor([1, 2, 1, 3, 4,
                                      5, 6, 7, 8, 9,
                                      10, 11, 12, 13,
                                      14, 15, 16],
                                      dtype=torch.long)

    

    assert torch.all(rna_seq_feat(good_seq) == good_seq_expected)
    assert rna_seq_feat.unfeaturize(good_seq_expected) == good_seq


def test_RNASequenceFeaturizer_bad_sequence(rna_seq_feat):
    bad_seq = "ACGT"

    with pytest.raises(ValueError, match=r"Unrecognized token 'T' for RNASequenceFeaturizer."):
        rna_seq_feat(bad_seq)


def test_ProteinSequenceFeaturizer(protein_seq_feat):
    good_seq = "ACADEFGHIKLMNPQRSTVWY-*"
    good_seq_expected = torch.tensor([1, 2, 1, 3, 4,
                                      5, 6, 7, 8, 9,
                                      10, 11, 12, 13,
                                      14, 15, 16, 17,
                                      18, 19, 20, 21,
                                      22],
                                      dtype=torch.long)

    assert torch.all(protein_seq_feat(good_seq) == good_seq_expected)
    assert protein_seq_feat.unfeaturize(good_seq_expected) == good_seq


def test_ProteinSequenceFeaturizer_bad_sequence(protein_seq_feat):
    bad_seq = "ACGTU"

    with pytest.raises(ValueError, match=r"Unrecognized token 'U' for ProteinSequenceFeaturizer."):
        protein_seq_feat(bad_seq)