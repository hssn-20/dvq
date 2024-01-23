from collections import Counter
import itertools

def _generate_all_permutations(kmer):
    nucleotides = ['A', 'T', 'G', 'C']
    return [''.join(p) for p in itertools.product(nucleotides, repeat=kmer)]

def _permutation_counts(sequence, permutations):
    if not isinstance(sequence, str):
        raise ValueError("Sequence must be a string")
    kmer = len(permutations[0])
    sequence = sequence.upper()
    sequence_counts = Counter(sequence[i:i+kmer] for i in range(len(sequence) - kmer + 1))
    return {permutation: sequence_counts.get(permutation, 0) for permutation in permutations}


def _validate_counts(counts):
    if not isinstance(counts, dict):
        raise TypeError("Counts must be a dictionary")

def _calculate_overlap(counts1, counts2):
    _validate_counts(counts1)
    _validate_counts(counts2)

    total1, total2 = sum(counts1.values()), sum(counts2.values())
    if total1 == 0 or total2 == 0:
        return 1 if total1 != total2 else 0

    overlap = 0
    for kmer in counts1.keys():
        count1 = counts1[kmer] / total1
        count2 = counts2.get(kmer, 0) / total2
        overlap += abs(count1 - count2)
    overlap = 1 - overlap
    return overlap

def permutation_overlap(sequence1, sequence2, kmer=6):
    if kmer <= 0:
        raise ValueError('kmer must be greater than 0')

    permutations = _generate_all_permutations(kmer)
    counts1 = _permutation_counts(sequence1, permutations)
    counts2 = _permutation_counts(sequence2, permutations)

    return _calculate_overlap(counts1, counts2)

# Rest of the functions remain unchanged

# Example call (ensure seqs_1 is a list of DNA sequence strings)
# permutation_overlap_comparison(seqs_1[:10], seqs_1[11:20], kmer=7)


def permutation_overlap_group(sequences:list, kmer= 7):
    """ Calculate the permutation overlap between all sequences in a list """
    if kmer <= 0:
        raise ValueError('kmer must be greater than 0')

    permutations = _generate_all_permutations(kmer)
    counts = [_permutation_counts(sequence, permutations) for sequence in sequences]

    overlap_groups = []
    for i, counts1 in enumerate(counts):
        overlap_group = []
        for j, counts2 in enumerate(counts):
            overlap = _calculate_overlap(counts1, counts2)
            overlap_group.append(overlap)
        overlap_groups.append(overlap_group)

    return overlap_groups
def permutation_overlap_comparison(sequences1: list, sequences2: list, kmer=7):
    """ Calculate the permutation overlap between all sequences in two lists """
    if kmer <= 0:
        raise ValueError('kmer must be greater than 0')

    permutations = _generate_all_permutations(kmer)
    counts_list1 = [_permutation_counts(sequence, permutations) for sequence in sequences1]
    counts_list2 = [_permutation_counts(sequence, permutations) for sequence in sequences2]

    overlap_groups = []
    for counts1 in counts_list1:
        overlap_group = []
        for counts2 in counts_list2:
            overlap = _calculate_overlap(counts1, counts2)
            overlap_group.append(overlap)
        overlap_groups.append(overlap_group)

    return overlap_groups
