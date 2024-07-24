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


# %%
# # test chase
# from datasets import load_dataset
# import pandas as pd
# from huggingface_hub import HfApi

# def main():
#     ds = load_dataset("DNA-LLM/virus_detailed_clean")

#     df = pd.DataFrame(ds['train'])

#     df_filter = df.groupby('family').head(5).reset_index(drop=True)

#     seqs = df_filter['seq'].tolist()

#     entropies = calculate_deng_entropies_multiprocess(seqs)

#     # df_lineage

#     df_lineage = pd.read_csv('data_testing/Viral Complexity - host_lineage.csv')

#     cleaned_lineage = []
#     for i in range(len(df_lineage)):
#         cleaned_data = {}
#     for m in range(11):
#         if m ==0:
#             continue
#         if df_lineage.iloc[i][f'rank_{m}'] =='species':
#             cleaned_data['species'] = df_lineage.iloc[i][f'name_{m}']
#         if df_lineage.iloc[i][f'rank_{m}'] =='genus':
#             cleaned_data['genus'] = df_lineage.iloc[i][f'name_{m}']
#         if df_lineage.iloc[i][f'rank_{m}'] =='family':
#             cleaned_data['family'] =df_lineage.iloc[i][f'name_{m}']
#         if df_lineage.iloc[i][f'rank_{m}'] =='order':
#             cleaned_data['order'] =df_lineage.iloc[i][f'name_{m}']
#         if df_lineage.iloc[i][f'rank_{m}'] =='class':
#             cleaned_data['class'] = df_lineage.iloc[i][f'name_{m}']
#         if df_lineage.iloc[i][f'rank_{m}'] =='phylum':
#             cleaned_data['phylum'] = df_lineage.iloc[i][f'name_{m}']
#         if df_lineage.iloc[i][f'rank_{m}'] =='kingdom':
#             cleaned_data['kingdom'] = df_lineage.iloc[i][f'name_{m}']
#         if df_lineage.iloc[i][f'rank_{m}'] =='superkingdom':
#             cleaned_data['superkingdom'] = df_lineage.iloc[i][f'name_{m}']
#         if df_lineage.iloc[i][f'rank_{m}'] =='clade':
#             cleaned_data['clade'] = df_lineage.iloc[i][f'name_{m}']
#     cleaned_lineage.append(cleaned_data)

#     # df_lineage

#     # df_lineage[i][f'rank_{m}'] =='species'

#     df_cleaned = pd.DataFrame(cleaned_lineage)

#     df_filter.to_parquet('data_testing/df_filtered.parquet')

#     # df_cleaned.phylum.value_counts()

#     df_filter.merge(df_lineage, left_on = 'host', right_on = 'name_1')

#     # df_filter

#     df_filter['entropy'] = entropies
#     df_filter.to_parquet('data_testing/viral_complexity_deng_entropy.parquet')

#     # deng_entropy_optimized(df_filter.iloc[39]['seq'])

#     # df_filter.iloc[39]

#     # %%
#     # upload
#     api = HfApi()
    
#     # api.create_repo(
#     #     repo_id="DNA-LLM/viral_complexity_deng_entropy",
#     #     repo_type="dataset",
#     #     private=True,  # set to True if you want the repository to be private
#     # )

#     # api.upload_file(
#     #     path_or_fileobj="data/viral_complexity_deng_entropy.parquet",
#     #     path_in_repo=f"data/viral_complexity_deng_entropy.parquet",
#     #     repo_id="DNA-LLM/viral_complexity_deng_entropy",
#     #     repo_type="dataset",
#     #     create_pr=False,
#     # )

# # %%
# if __name__ == "__main__":
#     main()