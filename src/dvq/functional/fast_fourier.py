# Fourier-based representation and comparison of biological sequences.
# Each sequence is encoded as four binary indicator signals (one per nucleotide),
# transformed via the Real Discrete Fourier Transform (rDFT) and summarised
# as a z-score normalised power spectrum. This captures the periodic structure
# of the sequence, such as CDR/framework alternation in antibody and TCR
# sequences.

import numpy as np
from typing import Literal

# Nucleotide channels for binary indicator encoding.
# Each channel produces a binary signal that is 1 where the nucleotide
# appears in the sequence and 0 everywhere else.
NUCLEOTIDE_CHARS = [ord('A'), ord('C'), ord('G'), ord('T')]

# Default number of frequency bins to interpolate spectra to before comparison.
# Using a power of 2 is conventional and covers the resolution needed for
# typical antibody/TCR sequence lengths (100-500 bp).
DEFAULT_SPECTRUM_BINS = 512


def _binary_indicator_encoding(seq: str) -> np.ndarray:
    """
    Encode a DNA sequence as a (4 x L) binary indicator matrix, where L is
    the sequence length and each row corresponds to one nucleotide channel
    (A, C, G, T). Entry [i, j] is 1 if position j is nucleotide i, else 0.

    Uses a vectorised NumPy implementation for performance on long sequences.
    Unknown characters (N, gaps) are silently assigned zero across all channels.

    Parameters:
    seq (str): The input DNA sequence.

    Returns:
    np.ndarray: A (4 x L) binary matrix of dtype float.
    """
    # Encode sequence as ASCII integer array for vectorised comparison
    arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    matrix = np.zeros((4, len(arr)), dtype=float)
    for i, char_code in enumerate(NUCLEOTIDE_CHARS):
        matrix[i] = (arr == char_code)
    return matrix


def _normalise_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalisation to a power spectrum (subtract mean, divide by
    standard deviation). This removes differences in overall signal amplitude
    caused by sequence length or nucleotide composition, retaining only the
    periodic structure (shape) of the spectrum.

    If the standard deviation is zero (flat spectrum, e.g. a single-nucleotide
    sequence), the spectrum is returned as all zeros rather than raising.

    Parameters:
    spectrum (np.ndarray): The raw power spectrum.

    Returns:
    np.ndarray: The z-score normalised power spectrum.
    """
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    if std == 0:
        return np.zeros_like(spectrum)
    return (spectrum - mean) / std


def _interpolate_spectrum(spectrum: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Interpolate a power spectrum to a fixed number of frequency bins.

    This rescales the frequency axis so that bin k represents the same
    physical frequency (k / n_bins) regardless of the original sequence
    length. This is essential for comparing spectra from sequences of
    different lengths — without it, the same bin index corresponds to
    different physical frequencies across sequences, making bin-by-bin
    comparison meaningless.

    For example, the periodicity-3 peak in coding sequences always sits at
    frequency 1/3, but at different bin indices depending on sequence length.
    Interpolating to a fixed axis aligns these peaks correctly.

    Parameters:
    spectrum (np.ndarray): The raw power spectrum to interpolate.
    n_bins (int): The target number of frequency bins.

    Returns:
    np.ndarray: The interpolated spectrum of length n_bins.
    """
    original_bins = np.linspace(0, 1, len(spectrum))
    target_bins = np.linspace(0, 1, n_bins)
    return np.interp(target_bins, original_bins, spectrum)


def power_spectrum(
    seq: str,
    normalise: bool = True,
    n_bins: int = DEFAULT_SPECTRUM_BINS
) -> np.ndarray:
    """
    Compute the power spectrum of a DNA sequence using the Real Discrete
    Fourier Transform (rDFT).

    Each nucleotide is encoded as a binary indicator signal. np.fft.rfft is
    applied to each of the four channels independently — exploiting the fact
    that the input signals are real-valued, which makes rfft both faster and
    more memory-efficient than the full FFT. The power spectra (squared
    magnitudes) of the four channels are summed into a single combined spectrum.

    The combined spectrum is then interpolated to n_bins frequency bins so
    that the output is always the same length regardless of input sequence
    length. This aligns the frequency axis across sequences, ensuring that
    the same bin index always represents the same physical frequency.

    Optionally, z-score normalisation is applied to remove amplitude differences
    and retain only the periodic structure of the sequence.

    Parameters:
    seq (str): The input DNA sequence.
    normalise (bool): Whether to apply z-score normalisation (default: True).
    n_bins (int): Number of frequency bins in the output spectrum (default: 512).

    Returns:
    np.ndarray: The power spectrum of length n_bins.

    Raises:
    ValueError: If the sequence is empty.
    """
    if not seq:
        raise ValueError("Sequence must not be empty.")

    # Vectorised binary encoding: shape (4, L)
    encoded = _binary_indicator_encoding(seq)

    # rfft on real-valued signals: returns L//2 + 1 non-redundant coefficients.
    # This is faster and more memory-efficient than fft + manual slicing.
    combined_power = np.zeros(len(seq) // 2 + 1, dtype=float)
    for channel in encoded:
        fft_coeffs = np.fft.rfft(channel)
        combined_power += np.abs(fft_coeffs) ** 2

    # Interpolate to fixed number of bins so output length is always n_bins,
    # and frequency bins are aligned across sequences of different lengths.
    combined_power = _interpolate_spectrum(combined_power, n_bins)

    if normalise:
        combined_power = _normalise_spectrum(combined_power)

    return combined_power


def spectral_distance(
    seq_P: str,
    seq_Q: str,
    metric: Literal["euclidean", "correlation"] = "euclidean",
    n_bins: int = DEFAULT_SPECTRUM_BINS
) -> float:
    """
    Compute the distance between two DNA sequences based on their power spectra.

    Both sequences are converted to z-score normalised power spectra of the
    same length (n_bins), with frequency axes aligned via interpolation.
    This ensures that bin-by-bin comparison is meaningful even when the input
    sequences differ in length.

    Two distance metrics are supported:
    - 'euclidean': L2 distance between the two spectra. Sensitive to differences
      in power at each individual frequency bin.
    - 'correlation': 1 minus the Pearson correlation between the two spectra.
      Sensitive to differences in the overall shape of the spectrum rather than
      absolute power values. Ranges from 0 (identical shape) to 2 (perfectly
      anti-correlated).

    Parameters:
    seq_P (str): The first DNA sequence.
    seq_Q (str): The second DNA sequence.
    metric (str): Distance metric, either 'euclidean' or 'correlation'
                  (default: 'euclidean').
    n_bins (int): Number of frequency bins to use for comparison (default: 512).

    Returns:
    float: The spectral distance between the two sequences.

    Raises:
    ValueError: If an unsupported metric is provided.
    ValueError: If either sequence is empty.
    """
    # Both spectra are interpolated to the same n_bins inside power_spectrum,
    # so no further alignment is needed here.
    sp = power_spectrum(seq_P, normalise=True, n_bins=n_bins)
    sq = power_spectrum(seq_Q, normalise=True, n_bins=n_bins)

    if metric == "euclidean":
        return float(np.linalg.norm(sp - sq))

    elif metric == "correlation":
        # Pearson correlation between the two spectra.
        # np.corrcoef returns a 2x2 matrix; [0,1] is the cross-correlation.
        corr = np.corrcoef(sp, sq)[0, 1]
        # Clip to [-1, 1] to guard against floating point noise
        corr = np.clip(corr, -1.0, 1.0)
        return float(1.0 - corr)

    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose 'euclidean' or 'correlation'."
        )