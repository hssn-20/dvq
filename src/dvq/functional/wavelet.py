# Wavelet-based representation and comparison of biological sequences.
# Each sequence is encoded as four binary indicator signals (one per nucleotide)
# and decomposed using the Discrete Wavelet Transform (DWT) with a Daubechies
# mother wavelet. The representation is a z-score normalised energy vector
# summarising the signal's activity at each decomposition level (scale).
import numpy as np
import pywt
from typing import Literal

# Nucleotide channels for binary indicator encoding
NUCLEOTIDE_CHARS = [ord('A'), ord('C'), ord('G'), ord('T')]

# Default mother wavelet: Daubechies order 4.
# Chosen for its smoothness and compact support, well suited to the gradual
# compositional transitions found in CDR and framework regions.
DEFAULT_WAVELET = 'db4'

# Default number of decomposition levels.
DEFAULT_N_LEVELS = 6

def _binary_indicator_encoding(seq: str) -> np.ndarray:
    """
    Encode a DNA sequence as a (4xL) binary indicator matrix, where L is
    the sequence length and each row corresponds to one nucleotide channel
    (A, C, G, T). Entry [i, j] is 1 if position j is nucleotide i, else 0.

    Uses a vectorised NumPy implementation for performance on long sequences.
    Unknown characters (N, gaps) are silently assigned zero across all channels.

    Parameters:
    seq (str): the input DNA sequence.

    Returns:
    np.ndarray: a (4xL) binary matrix of dtype float.
    """
    arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    matrix = np.zeros((4, len(arr)), dtype=float)
    for i, char_code in enumerate(NUCLEOTIDE_CHARS):
        matrix[i] = (arr == char_code)
    return matrix


def _level_energy(coefficients: np.ndarray) -> float:
    """
    Compute the energy of a set of wavelet coefficients as the sum of their
    squared values. 

    Parameters:
    coefficients (np.ndarray): wavelet coefficients at one decomposition level.

    Returns:
    float: the energy of the coefficients.
    """
    return float(np.sum(coefficients ** 2))


def _normalise_energy_vector(energy_vector: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalisation to an energy vector. Applied at the end of the pipeline so that the
    relative energy distribution across scales is preserved.

    If the standard deviation is zero (flat energy vector), the vector is
    returned as all zeros.

    Parameters:
    energy_vector (np.ndarray): the raw energy vector across decomposition levels.

    Returns:
    np.ndarray: the z-score normalised energy vector.
    """
    mean = np.mean(energy_vector)
    std = np.std(energy_vector)
    if std == 0:
        return np.zeros_like(energy_vector)
    return (energy_vector - mean) / std


def wavelet_energy(
    seq: str,
    wavelet: str = DEFAULT_WAVELET,
    n_levels: int = DEFAULT_N_LEVELS,
    normalise: bool = True
) -> np.ndarray:
    """
    Compute the wavelet energy vector of a DNA sequence using the Discrete
    Wavelet Transform (DWT).

    The sequence is encoded as four binary indicator signals.
    The DWT is applied to each channel independently using the specified mother
    wavelet. At each decomposition level, the energy is computed for the detail
    coefficients. The energy of the final approximation
    coefficients is also included as the last element, capturing the residual
    low-frequency trend of the sequence.

    The energies are summed across the four nucleotide channels and optionally
    z-score normalised, producing a fixed-size vector of length n_levels + 1
    regardless of the input sequence length.

    If a sequence is too short to support the requested number of decomposition
    levels, the missing levels are zero-padded at the coarser end of the vector.
    This preserves vector length consistency without discarding information from
    sequences that do support those levels.

    Parameters:
    seq (str): the input DNA sequence.
    wavelet (str): the mother wavelet to use (default: 'db4'). Any wavelet
                   supported by PyWavelets is valid (e.g. 'db2', 'db8', 'haar').
    n_levels (int): number of decomposition levels (default: 6). The output
                    vector will have length n_levels + 1 (levels + approximation).
    normalise (bool): whether to apply z-score normalisation (default: True).

    Returns:
    np.ndarray: the (normalised) energy vector of length n_levels + 1.

    Raises:
    ValueError: if the sequence is empty.
    """
    if not seq:
        raise ValueError("Sequence must not be empty.")

    encoded = _binary_indicator_encoding(seq)

    # Energy vector: n_levels detail energies + 1 approximation energy
    # Initialised to zeros so short sequences are automatically zero-padded
    # at levels they cannot reach.
    combined_energy = np.zeros(n_levels + 1, dtype=float)

    for channel in encoded:
        # Compute the maximum number of levels this sequence can support
        max_levels = pywt.dwt_max_level(len(channel), wavelet)
        actual_levels = min(n_levels, max_levels)

        # wavedec returns [approximation, detail_N, detail_N-1, ..., detail_1]
        # Request actual_levels to avoid requesting more than the sequence supports
        coeffs = pywt.wavedec(channel, wavelet, level=actual_levels)

        # coeffs[0] is the approximation, coeffs[1:] are detail levels
        # Detail coefficients run from coarsest (level N) to finest (level 1)
        # We store them finest-first (index 0 = finest scale = level 1)
        details = coeffs[1:][::-1]  # reverse to finest-first order
        approximation = coeffs[0]

        # Accumulate detail energies into the combined vector
        for level_idx, detail_coeffs in enumerate(details):
            combined_energy[level_idx] += _level_energy(detail_coeffs)

        # Accumulate approximation energy into the last position
        combined_energy[n_levels] += _level_energy(approximation)

    if normalise:
        combined_energy = _normalise_energy_vector(combined_energy)

    return combined_energy


def wavelet_distance(
    seq_P: str,
    seq_Q: str,
    metric: Literal["euclidean", "correlation"] = "euclidean",
    wavelet: str = DEFAULT_WAVELET,
    n_levels: int = DEFAULT_N_LEVELS
) -> float:
    """
    Compute the distance between two DNA sequences based on their wavelet
    energy vectors.

    Both sequences are converted to z-score normalised energy vectors of the
    same fixed length (n_levels + 1). Since the output of wavelet_energy is
    always fixed-size regardless of input sequence length, no further alignment
    is needed before comparison.

    Two distance metrics are supported:
    - euclidean: L2 distance between the two energy vectors. Sensitive to
      differences in energy at each individual decomposition level.
    - correlation: 1 minus the Pearson correlation between the two energy
      vectors. Sensitive to differences in the overall distribution of energy
      across scales — e.g. whether one sequence concentrates energy at fine
      scales (CDR-rich) vs coarse scales (framework-rich) relative to the other.
      Ranges from 0 (identical profile) to 2 (perfectly anti-correlated).

    Parameters:
    seq_P (str): the first DNA sequence.
    seq_Q (str): the second DNA sequence.
    metric (str): distance metric, either euclidean or correlation
                  (default: 'euclidean').
    wavelet (str): the mother wavelet to use (default: db4).
    n_levels (int): number of decomposition levels (default: 6).

    Returns:
    float: the wavelet distance between the two sequences.

    Raises:
    ValueError: if an unsupported metric is provided.
    ValueError: if either sequence is empty.
    """
    ep = wavelet_energy(seq_P, wavelet=wavelet, n_levels=n_levels, normalise=True)
    eq = wavelet_energy(seq_Q, wavelet=wavelet, n_levels=n_levels, normalise=True)

    if metric == "euclidean":
        return float(np.linalg.norm(ep - eq))

    elif metric == "correlation":
        corr = np.corrcoef(ep, eq)[0, 1]
        corr = np.clip(corr, -1.0, 1.0)
        return float(1.0 - corr)

    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose 'euclidean' or 'correlation'."
        )