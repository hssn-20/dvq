import numpy as np
from typing import List, Dict
from .KL_divergence import calculate_probabilities

def js_divergence(seq_P: str, seq_Q:str) -> float:
    """
    Calculate the JS divergence between two probability distributions.
    
    Parameters:
    seq_P (str): first sequence.
    seq_Q (str): second sequence.
    
    Returns:
    float: The JS divergence.
    """
    P = calculate_probabilities(seq_P)
    Q = calculate_probabilities(seq_Q)

    keys = set(P.keys()) | set(Q.keys())
    M = {k: 0.5 * P.get(k, 0.0) + 0.5 * Q.get(k, 0.0) for k in keys} 

    # KL(P || M)
    kl_pm = 0.0
    for chunk, p_prob in P.items():
        m_prob = M.get(chunk)
        kl_pm += p_prob * np.log2(p_prob / m_prob)
    
    # KL(Q || M)
    kl_qm = 0.0
    for chunk, q_prob in Q.items():
        m_prob = M.get(chunk)
        kl_qm += q_prob * np.log2(q_prob / m_prob)
    
    js_div = 0.5 * (kl_pm + kl_qm)
    return js_div

