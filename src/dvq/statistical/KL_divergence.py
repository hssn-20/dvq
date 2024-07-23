import numpy as np

from typing import List, Dict

def calculate_probabilities(seq: str, chunk_size: int = 10) -> Dict[str, float]:
    """
    Calculate the probability distribution of chunks in a sequence.
    
    Parameters:
    seq (str): The input sequence.
    chunk_size (int): The size of each chunk.
    
    Returns:
    Dict[str, float]: A dictionary with chunks as keys and their probabilities as values.
    """
    seq_len = len(seq)
    chunks = [seq[i:i+chunk_size] for i in range(len(seq) - chunk_size + 1)]
    chunk_counter = Counter(chunks)
    probabilities = {chunk: count / seq_len for chunk, count in chunk_counter.items()}
    return probabilities


def kl_divergence(P: Dict[str, float], Q: Dict[str, float]) -> float:
    """
    Calculate the KL divergence between two probability distributions.
    
    Parameters:
    P (Dict[str, float]): The first probability distribution.
    Q (Dict[str, float]): The second probability distribution.
    
    Returns:
    float: The KL divergence.
    """
    kl_div = 0.0
    for chunk, p_prob in P.items():
        q_prob = Q.get(chunk, 1e-10)  # To avoid division by zero, use a small value for missing chunks in Q
        kl_div += p_prob * np.log2(p_prob / q_prob)
    return kl_div


def kl_divergence(seq_P: str, seq_Q: str) -> float:
    """
    Calculate the KL divergence between two probability distributions.
    
    Parameters:
    seq_P (str): first sequence.
    seq_Q (str): second sequence.
    
    Returns:
    float: The KL divergence.
    """
    P = calculate_probabilities(seq_P)
    Q = calculate_probabilities(seq_Q)

    kl_div = 0.0
    for chunk, p_prob in P.items():
        q_prob = Q.get(chunk, 1e-10)  # To avoid division by zero, use a small value for missing chunks in Q
        kl_div += p_prob * np.log2(p_prob / q_prob)
    return kl_div


# %%
# test case

seq_p = "ACGTACGTACGTACGTACGT"
seq_q = "ACGTACGTACGTACGTGCGC"

P = calculate_probabilities(seq_p)
Q = calculate_probabilities(seq_q)

kl_div = kl_divergence(P, Q)
print(f"KL Divergence: {kl_div}")