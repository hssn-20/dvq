from collections import defaultdict
from sourmash import MinHash
from tqdm import tqdm 
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def _create_mini_hash_of_a_sequence(seq: str, minihash: MinHash) -> MinHash:
    """Create a MinHash of a sequence (or list of sequences)."""
    # If seq is a list of sequences, add each string to the minihash.
    # If seq is a single string, this loop will just run once.
    if isinstance(seq, list):
        for s in seq:
            minihash.add_sequence(s)
    else:
        minihash.add_sequence(seq)
    return minihash

def _compare_two_sequences_and_return_similarity(
    seq: str, seq2: str, k: int, n: int
) -> float:
    """Calculate similarity (Jaccard) of two sequences with given k and number of hashes."""
    mh1 = MinHash(n=n, ksize=k)
    mh2 = MinHash(n=n, ksize=k)
    mh1 = _create_mini_hash_of_a_sequence(seq, mh1)
    mh2 = _create_mini_hash_of_a_sequence(seq2, mh2)
    similarity = round(mh1.similarity(mh2), 5)
    return similarity

def average_kmer_jaccard_similarity(
    seq,
    seq2,
    number_of_hashes: int = 20000,
    k_sizes: list = [1, 3, 7, 20]
) -> float:
    """
    Calculate the average Jaccard similarity for two sequence inputs
    across multiple k-mer sizes. Also logs the individual similarity
    for each k-mer size.
    """
    # Ensure seq and seq2 are lists of strings
    if not isinstance(seq, list):
        seq = [seq]
    if not isinstance(seq2, list):
        seq2 = [seq2]

    similarities = []
    for k in tqdm(k_sizes):
        sim = _compare_two_sequences_and_return_similarity(seq, seq2, k, number_of_hashes)
        # Log (or print) the similarity for this k-mer size
        logging.info(f"K = {k}, similarity = {sim:.5f}")
        similarities.append(sim)

    # Compute and return the average similarity
    avg_similarity = round(np.mean(similarities), 3)
    logging.info(f"Average similarity across k={k_sizes} is {avg_similarity:.3f}")
    return avg_similarity
