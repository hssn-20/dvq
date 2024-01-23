import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')

# Mapping of nucleotides to float coordinates
mapping_easy = {
    'A': np.array([0.5, -0.8660254037844386]),
    'T': np.array([0.5, 0.8660254037844386]),
    'G': np.array([0.8660254037844386, -0.5]),
    'C': np.array([0.8660254037844386, 0.5]),
    'N': np.array([0, 0])
}

# Function to convert a DNA sequence to a list of coordinates
def _dna_to_coordinates(dna_sequence, mapping):
    dna_sequence = dna_sequence.upper()
    coordinates = np.array([mapping.get(nucleotide, mapping['N']) for nucleotide in dna_sequence])
    return coordinates

# Function to create the cumulative sum of a list of coordinates
def _get_cumulative_coords(mapped_coords):
    cumulative_coords = np.cumsum(mapped_coords, axis=0)
    return cumulative_coords

# Function to take a list of DNA sequences and plot them in a single figure
def plot_2d_sequences(dna_sequences, mapping=mapping_easy, single_sequence=False):
    fig, ax = plt.subplots()
    if single_sequence:
        dna_sequences = [dna_sequences]
    for dna_sequence in dna_sequences:
        mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
        cumulative_coords = _get_cumulative_coords(mapped_coords)
        ax.plot(*cumulative_coords.T)
    plt.show()

# Function to plot a comparison of DNA sequences
def plot_2d_comparison(dna_sequences_grouped, labels, mapping=mapping_easy):
    fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(dna_sequences_grouped)))
    for count, (dna_sequences, color) in enumerate(zip(dna_sequences_grouped, colors)):
        for dna_sequence in dna_sequences:
            mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
            cumulative_coords = _get_cumulative_coords(mapped_coords)
            ax.plot(*cumulative_coords.T, color=color, label=labels[count])
    # Only show unique labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()
