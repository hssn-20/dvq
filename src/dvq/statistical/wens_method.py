import numpy as np

# Constants
WEIGHTS = {'0100': 1/6, '0101': 2/6, '1100' : 3/6, '0110':3/6, '1101': 4/6, '1110': 5/6,'0111':5/6, '1111': 6/6}
LOWEST_LENGTH = 5000


def _get_subsequences(sequence):
    return {nuc: [i+1 for i, x in enumerate(sequence) if x == nuc] for nuc in 'ACTG'}


def _calculate_coordinates_fixed(subsequence, L=LOWEST_LENGTH):
    return [((2 * np.pi / (L - 1)) * (K-1), np.sqrt((2 * np.pi / (L - 1)) * (K-1))) for K in subsequence]


def _calculate_weighting_full(sequence, WEIGHTS, L=LOWEST_LENGTH, E=0.0375):
    weightings = [0]
    for i in range(1, len(sequence) - 1):
        if i < len(sequence) - 2:
            subsequence = sequence[i-1:i+3]
            comparison_pattern = f"{'1' if subsequence[0] == subsequence[1] else '0'}1{'1' if subsequence[2] == subsequence[1] else '0'}{'1' if subsequence[3] == subsequence[1] else '0'}"
            weight = WEIGHTS.get(comparison_pattern, 0)
            weight = weight * E if i > L else weight
        else:
            weight = 0
        weightings.append(weight)
    weightings.append(0)
    return weightings


def _centre_of_mass(polar_coordinates, weightings):
    x, y = _calculate_standard_coordinates(polar_coordinates)
    return sum(weightings[i] * ((x[i] - (x[i]*weightings[i]))**2 + (y[i] - y[i]*weightings[i])**2) for i in range(len(x)))


def _normalised_moment_of_inertia(polar_coordinates, weightings):
    moment = _centre_of_mass(polar_coordinates, weightings)
    return np.sqrt(moment / sum(weightings))


def _calculate_standard_coordinates(polar_coordinates):
    return [rho * np.cos(theta) for theta, rho in polar_coordinates], [rho * np.sin(theta) for theta, rho in polar_coordinates]


def _moments_of_inertia(polar_coordinates, weightings):
    return [_normalised_moment_of_inertia(indices, weightings) for subsequence, indices in polar_coordinates.items()]


def moment_of_inertia(sequence, WEIGHTS=WEIGHTS, L=5000, E=0.0375):
    subsequences = _get_subsequences(sequence)
    polar_coordinates = {subsequence: _calculate_coordinates_fixed(indices, len(sequence)) for subsequence, indices in subsequences.items()}
    weightings = _calculate_weighting_full(sequence, WEIGHTS, L=L, E=E)
    return _moments_of_inertia(polar_coordinates, weightings)


def similarity_wen(sequence1, sequence2, WEIGHTS=WEIGHTS, L=5000, E=0.0375):
    inertia1 = moment_of_inertia(sequence1, WEIGHTS, L=L, E=E)
    inertia2 = moment_of_inertia(sequence2, WEIGHTS, L=L, E=E)
    similarity = np.sqrt(sum((x - y)**2 for x, y in zip(inertia1, inertia2)))
    return similarity
