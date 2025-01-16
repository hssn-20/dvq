# Orignial function written by Hassan Ahmed Hassan, following: 
# TODO: Add paper doi
# %%
import numpy as np
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple, List

def get_possibilities(seq: Tuple[str, ...]) -> int:
    """
    Calculate the number of unique possible combinations for a given sequence.
    
    Parameters:
    seq (Tuple[str, ...]): The input sequence as a tuple of characters.
    
    Returns:
    int: The number of unique possible combinations.
    """
    if len(seq) == 1:
        return 1

    # Convert seq to a sorted tuple for consistent behavior
    seq = tuple(sorted(seq))

    # Generate unique combinations
    unique_combs = set()
    for i in range(1, len(seq)):
        unique_combs.update(''.join(sorted(comb)) for comb in combinations(seq, i))

    # Filter combinations
    filtered_combs = set()
    for comb in unique_combs:
        if not any(set(item).issubset(set(comb)) for item in filtered_combs):
            filtered_combs.add(comb)

    return len(filtered_combs)

def dedupe(s: str) -> str:
    """
    Remove duplicate characters and sort the remaining characters in a string.
    
    Parameters:
    s (str): The input string.
    
    Returns:
    str: A string with unique and sorted characters.
    """
    return ''.join(sorted(set(s)))

def generalised_version(seq: str, chunk_size: int = 10) -> float:
    """
    Calculate the generalised entropy for a given sequence using a specified chunk size.
    
    Parameters:
    seq (str): The input sequence.
    chunk_size (int): The size of each chunk.
    
    Returns:
    float: The calculated entropy.
    """
    seq = seq[:10_000]
    seq_len = len(seq)

    # Use sliding window for chunk creation
    chunks = [tuple(seq[i:i+chunk_size]) for i in range(len(seq) - chunk_size + 1)]
    unique_chunks = list(set(chunks))

    # Use Counter for efficient counting
    chunk_counter = Counter(chunks)

    # Calculate chunk percentages and possibilities
    chunk_percentages = np.array([chunk_counter[chunk] / seq_len for chunk in unique_chunks])
    chunk_possibilities = np.array([get_possibilities(chunk) or 1 for chunk in unique_chunks])

    # Handle single element case
    if len(set(seq)) == 1:
        return -np.log2(1 / seq_len)

    # Vectorized entropy calculation
    mask = chunk_possibilities > 1000
    entropy_high = np.sum(chunk_percentages[mask] * (np.log2(chunk_percentages[mask]/ chunk_possibilities[mask])))
    entropy_low = np.sum(chunk_percentages[~mask] * np.log2(chunk_percentages[~mask] / chunk_possibilities[~mask]))
    
    return -(entropy_high + entropy_low)

def process_sequence(seq: str) -> float:
    """
    Process a single sequence to calculate its generalised entropy.
    
    Parameters:
    seq (str): The input sequence.
    
    Returns:
    float: The calculated entropy.
    """
    return generalised_version(seq)

def calculate_deng_entropies_multiprocess(seqs: List[str], num_cores: int = cpu_count()) -> List[float]:
    """
    Calculate the generalised entropy for multiple sequences using multiprocessing.
    
    Parameters:
    seqs (List[str]): The list of input sequences.
    num_cores (int): The number of cores to use for multiprocessing.
    
    Returns:
    List[float]: A list of calculated entropies for each sequence.
    """
    with Pool(num_cores) as pool:
        entropies = list(tqdm(pool.imap(process_sequence, seqs), total=len(seqs)))
    return entropies


# experimental 
def deng_KL_divergence(seq_P: str, seq_Q: str, chunk_size: int = 10) -> float:
    """
    Compute a KL divergence-like metric using Deng entropy for two sequences.
    
    Parameters:
    seq_P (str): The first sequence.
    seq_Q (str): The second sequence.
    chunk_size (int): Chunk size for calculating Deng entropy (default: 10).
    
    Returns:
    float: The computed KL divergence-like metric.
    """
    # Compute Deng entropy for both sequences
    deng_entropy_P = generalised_version(seq_P, chunk_size)
    deng_entropy_Q = generalised_version(seq_Q, chunk_size)

    # Avoid division by zero or log of zero
    if deng_entropy_P == 0 or deng_entropy_Q == 0:
        raise ValueError("Deng entropy is zero for one or both sequences, cannot compute KL divergence.")

    # Compute the KL divergence-like metric
    deng_kl_div = deng_entropy_P * np.log2(deng_entropy_P / deng_entropy_Q)
    
    return deng_kl_div
