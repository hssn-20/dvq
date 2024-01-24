import itertools
# Function to generate all the possible DNA sequences of a given length
def _generate_all_permutation(length):
    if length == 0:
        raise ValueError('Length must be greater than 0')
    else:
        nucleotides = ['A', 'T', 'G', 'C']
        permutations = [''.join(p) for p in itertools.product(nucleotides, repeat=length)]
        return permutations


# Function to calculate the count how many times each permutations appears anywhere in a string
def _permutation_counts(sequence, permutations):
    counts = [sequence.count(permutation) for permutation in permutations]
    return counts

