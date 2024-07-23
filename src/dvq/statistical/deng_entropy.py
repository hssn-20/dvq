# Orignial function written by Hassan Ahmed Hassan, following: 
# TODO: Add paper doi
import numpy as np
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def get_possiblities(seq):
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

def dedupe(s):
    return ''.join(sorted(set(s)))

def generalised_version(seq, chunk_size=10):
    seq = seq[:10_000]
    seq_len = len(seq)

    # Use sliding window for chunk creation
    chunks = [tuple(seq[i:i+chunk_size]) for i in range(len(seq) - chunk_size + 1)]
    unique_chunks = list(set(chunks))

    # Use Counter for efficient counting
    chunk_counter = Counter(chunks)

    # Calculate chunk percentages and possibilities
    chunk_percentages = np.array([chunk_counter[chunk] / seq_len for chunk in unique_chunks])
    chunk_possibilities = np.array([get_possiblities(chunk) or 1 for chunk in unique_chunks])

    # Handle single element case
    if len(set(seq)) == 1:
        return -np.log2(1 / seq_len)

    # Vectorized entropy calculation
    mask = chunk_possibilities > 1000
    entropy_high = np.sum(chunk_percentages[mask] * (np.log2(chunk_percentages[mask]/ chunk_possibilities[mask] )))
    entropy_low = np.sum(chunk_percentages[~mask] * np.log2(chunk_percentages[~mask] / (chunk_possibilities[~mask])))
    print( -(entropy_high + entropy_low))
    return -(entropy_high + entropy_low)



def process_sequence(seq):
    return generalised_version(seq)

def calculate_deng_entropies_multiprocess(
        seqs, 
        num_cores = cpu_count()
    ):
    with Pool(num_cores) as pool:
        entropies = list(tqdm(pool.imap(process_sequence, seqs), total=len(seqs)))
    return entropies

seqs = df_filter['seq'].tolist()
entropies = calculate_entropies(seqs)