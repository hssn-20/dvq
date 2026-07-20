import pytest
import numpy as np
from dvq.statistical.wasserstein_distance import wasserstein_distance, wasserstein_distance_matrix


# ── Shared fixtures ────────────────────────────────────────────────────────────

SEQ_A = "ACGT" * 1000
SEQ_B = "GCTA" * 1000
SEQ_ALL_A = "A" * 100
SEQ_ALL_T = "T" * 100


# ── Mathematical properties ────────────────────────────────────────────────────

def test_wasserstein_identity():
    """
    Identity: wasserstein(seq, seq) should be 0.
    A sequence compared to itself requires zero transport cost,
    since both distributions are identical.
    """
    assert np.isclose(wasserstein_distance(SEQ_A, SEQ_A), 0.0)


def test_wasserstein_symmetry():
    """
    Symmetry: wasserstein(seq_1, seq_2) == wasserstein(seq_2, seq_1).
    The EMD is a true metric — direction of comparison should not matter.
    """
    assert np.isclose(
        wasserstein_distance(SEQ_A, SEQ_B),
        wasserstein_distance(SEQ_B, SEQ_A)
    ), "Wasserstein distance is not symmetric."


def test_wasserstein_non_negative():
    """
    Non-negativity: wasserstein(seq_1, seq_2) >= 0 for all sequences.
    Transport cost cannot be negative.
    """
    assert wasserstein_distance(SEQ_A, SEQ_B) >= 0.0
    

def test_wasserstein_triangle_inequality():
    """
    Triangle inequality: wasserstein(A, C) <= wasserstein(A, B) + wasserstein(B, C).
    As a proper metric, the Wasserstein distance must satisfy this property.
    """
    SEQ_C = "TTTT" * 1000
    d_ab = wasserstein_distance(SEQ_A, SEQ_B)
    d_bc = wasserstein_distance(SEQ_B, SEQ_C)
    d_ac = wasserstein_distance(SEQ_A, SEQ_C)
    assert d_ac <= d_ab + d_bc + 1e-6, (
        f"Triangle inequality violated: d(A,C)={d_ac} > d(A,B)+d(B,C)={d_ab+d_bc}"
    )


# ── Sensitivity ────────────────────────────────────────────────────────────────

def test_wasserstein_similar_sequences_lower_than_different():
    """
    Sensitivity: sequences that differ by one nucleotide should yield a lower
    Wasserstein distance than sequences that are completely different.
    Verifies that the Hamming ground metric is doing meaningful work.
    """
    base = "ACGTACGT" * 500
    one_substitution = base[:-1] + ("T" if base[-1] != "T" else "A")
    completely_different = "TTTT" * 1000

    d_close = wasserstein_distance(base, one_substitution)
    d_far = wasserstein_distance(base, completely_different)

    assert d_close < d_far, (
        f"Expected d(base, one_substitution)={d_close} < d(base, different)={d_far}"
    )


def test_wasserstein_identical_composition_different_order():
    """
    Composition sensitivity: two sequences with the same nucleotide composition
    but different arrangement should have a low but non-zero distance,
    since their k-mer frequency profiles differ.
    """
    seq_1 = "ACGT" * 500
    seq_2 = "AACCGGTT" * 250  # same A,C,G,T counts, different local structure
    d = wasserstein_distance(seq_1, seq_2)
    assert d >= 0.0
    assert not np.isclose(d, 0.0), (
        "Expected non-zero distance for sequences with different k-mer profiles."
    )


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_wasserstein_single_nucleotide_sequences():
    """
    Edge case: sequences composed of a single repeated nucleotide.
    AAAAA... vs TTTTT... should produce a non-zero, finite distance.
    """
    d = wasserstein_distance(SEQ_ALL_A, SEQ_ALL_T)
    assert d > 0.0
    assert np.isfinite(d), "Distance should be finite for single-nucleotide sequences."


def test_wasserstein_sequence_too_short_raises():
    """
    Edge case: a sequence shorter than k should raise a ValueError,
    since no k-mers can be extracted.
    """
    with pytest.raises(ValueError):
        wasserstein_distance("ACG", "ACGT" * 100, k=7)


def test_wasserstein_different_k_values():
    """
    Parameter sensitivity: distance values should vary with k, since larger k
    captures more local context and produces a finer-grained distribution.
    Results should always be non-negative and finite regardless of k.
    """
    for k in [3, 5, 7]:
        d = wasserstein_distance(SEQ_A, SEQ_B, k=k)
        assert d >= 0.0, f"Negative distance for k={k}"
        assert np.isfinite(d), f"Non-finite distance for k={k}"


def test_wasserstein_max_kmers_does_not_change_sign():
    """
    Stability: reducing max_kmers (vocabulary truncation) should not
    change the sign or finiteness of the result, even if the exact
    value shifts due to renormalisation.
    """
    d_full = wasserstein_distance(SEQ_A, SEQ_B, max_kmers=512)
    d_trimmed = wasserstein_distance(SEQ_A, SEQ_B, max_kmers=64)
    assert d_full >= 0.0
    assert d_trimmed >= 0.0
    assert np.isfinite(d_full)
    assert np.isfinite(d_trimmed)


# ── Distance matrix ────────────────────────────────────────────────────────────

def test_wasserstein_matrix_shape():
    """
    Matrix shape: the pairwise distance matrix for n sequences
    should be (n x n).
    """
    sequences = [SEQ_A, SEQ_B, SEQ_ALL_A]
    matrix = wasserstein_distance_matrix(sequences)
    assert matrix.shape == (3, 3), f"Expected shape (3,3), got {matrix.shape}"


def test_wasserstein_matrix_diagonal_is_zero():
    """
    Matrix identity: diagonal entries (self-distance) should all be zero.
    """
    sequences = [SEQ_A, SEQ_B, SEQ_ALL_A]
    matrix = wasserstein_distance_matrix(sequences)
    assert np.allclose(np.diag(matrix), 0.0), "Diagonal of distance matrix should be zero."


def test_wasserstein_matrix_symmetry():
    """
    Matrix symmetry: the distance matrix should be symmetric,
    i.e. matrix[i,j] == matrix[j,i] for all i, j.
    """
    sequences = [SEQ_A, SEQ_B, SEQ_ALL_A]
    matrix = wasserstein_distance_matrix(sequences)
    assert np.allclose(matrix, matrix.T), "Distance matrix is not symmetric."


def test_wasserstein_matrix_non_negative():
    """
    Matrix non-negativity: all entries in the distance matrix should be >= 0.
    """
    sequences = [SEQ_A, SEQ_B, SEQ_ALL_A]
    matrix = wasserstein_distance_matrix(sequences)
    assert np.all(matrix >= 0.0), "Distance matrix contains negative values."