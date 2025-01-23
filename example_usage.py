# %%
# Example usage for all scripts
from src.dvq.statistical import (
    average_kmer_jaccard_similarity,
    similarity_wen,
    moment_of_inertia,
    persistence_homology,
    compare_persistence_homology,
    denq_entropy_generalised,
    calculate_deng_entropies_multiprocess,
    deng_KL_divergence,
    kl_divergence
)


# %%
def main():
    # Example sequences
    seq_1 = "ACGT" * 1000
    seq_2 = "GCTA" * 1000
    seqs = [seq_1, seq_2, "TTAA" * 1000]

    # Test Deng Entropy
    print("\n--- Testing Deng Entropy ---")
    entropy = denq_entropy_generalised(seq_1)
    print(f"Deng entropy for sequence: {entropy}")

    entropies = calculate_deng_entropies_multiprocess(seqs)
    print(f"Entropies for sequences: {entropies}")

    # Test Deng KL Divergence
    print("\n--- Testing Deng KL Divergence ---")
    kl_div = deng_KL_divergence(seq_1, seq_2)
    print(f"Deng KL divergence: {kl_div}")

    # Test Standard KL Divergence
    print("\n--- Testing Standard KL Divergence ---")
    kl_div_standard = kl_divergence(seq_1, seq_2)
    print(f"Standard KL divergence: {kl_div_standard}")

    # Test Wen's Method
    print("\n--- Testing Wen's method ---")
    similarity = similarity_wen(seq_1, seq_2)
    print(f"Similarity using Wen's method: {similarity}")

    inertia = moment_of_inertia(seq_1)
    print(f"Moment of inertia for sequence: {inertia}")

    # Test Persistent Homology
    print("\n--- Testing Persistent Homology ---")
    persistence = persistence_homology(seq_1, plot=True)
    print(f"Persistent homology for sequence: {persistence}")

    homology_distance = compare_persistence_homology(seq_1, seq_2)
    print(f"Persistent homology distance: {homology_distance}")

    # Test k-mer functions
    print("\n--- Testing k-mer functions ---")
    overlap_group = average_kmer_jaccard_similarity(seqs, seqs)
    print(f"Self similarity: {overlap_group}")

    overlap_comparison = average_kmer_jaccard_similarity(seq_1, seq_2)
    print(f"Kmer similarity: {overlap_comparison}")

    return

if __name__ == "__main__":
    main()


## TODO: 
# 
# Add features:
# JS divergence, 
# Wasserstein distance