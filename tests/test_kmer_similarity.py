import pytest
from src.dvq.statistical.kmer_representation import average_kmer_jaccard_similarity

def test_identical_sequences():
    """Test that identical sequences yield a high Jaccard similarity."""
    seq1 = "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
    seq2 = "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
    similarity = average_kmer_jaccard_similarity(seq1, seq2)
    
    # For identical short sequences, the similarity across
    # multiple k-sizes should typically be close to 1.0
    # but might be slightly less, depending on how MinHash
    # collision or sampling is done. 
    # We use an assertion that it should be at least 0.9.
    assert similarity >= 0.9, f"Expected similarity >= 0.9, got {similarity}"

def test_completely_different_sequences():
    """Test that completely different sequences yield a low Jaccard similarity."""
    seq1 = "AAAAAAAA"  # all As
    seq2 = "CCCCCCCC"  # all Cs
    similarity = average_kmer_jaccard_similarity(seq1, seq2)
    
    # If the sequences differ completely, the similarity
    # should be close to 0. 
    assert similarity <= 0.1, f"Expected similarity <= 0.1, got {similarity}"

def test_partial_overlap_sequences():
    """Test that partially overlapping sequences yield a moderate Jaccard similarity."""
    # Here we create two sequences that share some overlap
    seq1 = "ATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTACATGCGTAC"
    seq2 = "ATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGACATGCGGAC"
    similarity = average_kmer_jaccard_similarity(seq1, seq2, k_sizes=[3])
    
    # These two sequences differ only in the middle base 
    # (T vs G at position 6). Expect a moderate similarity.
    # The exact value may vary based on hashing collisions 
    # and chosen k-mer sizes, but we can check it’s not too high or too low.
    assert 0.4 <= similarity <= 0.99, f"Expected similarity between 0.4 and 0.99, got {similarity}"

def test_list_of_sequences():
    """
    If your code supports passing lists of sequences, 
    test that it handles them gracefully.
    """
    seq1_list = ["ATGC", "AACC"]
    seq2_list = ["ATGC", "GGGG"] 
    similarity = average_kmer_jaccard_similarity(seq1_list, seq2_list)

    # Expect the similarity to not be extremely high because
    # the second list has 'GGGG' which doesn't appear in seq1_list.
    # But there's partial overlap on "ATGC".
    assert 0.1 <= similarity <= 0.9, f"Expected similarity between 0.1 and 0.9, got {similarity}"

