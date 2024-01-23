## 2D representation of DNA sequences

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')


# mapping of nucleotides to float coordinates
mapping_easy = {
    'A' : (0.5, -0.8660254037844386),
    'T' : (0.5, 0.8660254037844386),
    'G' : (0.8660254037844386, -0.5),
    'C' : (0.8660254037844386, 0.5)
    'N' : (0, 0)

}

# function to convert a DNA sequence to a list of coordinates
def _dna_to_coordinates(dna_sequence, mapping):
    coordinates = []
    for nucleotide in list(dna_sequence):
        nucleotide = str(nucleotide).upper()
        try:
            coordinates.append(mapping[nucleotide])
        except KeyError:
            coordinates.append(mapping['N'])
    return coordinates

# function to create the commulative sum of a list of coordinates
def _get_cumulative_coords(mapped_coords):
    cumulative_coords = []
    x = 0
    y = 0
    for coord in mapped_coords:
        x += coord[0]
        y += coord[1]
        cumulative_coords.append((x, y))
    return cumulative_coords

# function to take a list of DNA sequences and plot them in a single figure
def plot_2d_sequences(dna_sequences:list,mapping=mapping_easy, single_sequence=False):
    fig, ax = plt.subplots()
    if single_sequence:
        dna_sequences = [dna_sequences]
    for dna_sequence in dna_sequences:
        mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
        cumulative_coords = _get_cumulative_coords(mapped_coords)
        x, y = zip(*cumulative_coords)
        ax.plot(x, y)
    plt.show()

def plot_2d_comparison(dna_sequences_grouped:list, labels:list, mapping=mapping_easy):
    """ function to take a list of lists of DNA sequences and plot them in a single figure, with each list of DNA sequences having a different color.
    """
    fig, ax = plt.subplots()
    count = 0
    no_of_groups = len(dna_sequences_grouped)
    colors = plt.cm.rainbow(np.linspace(0, 1, no_of_groups))
    for dna_sequences in dna_sequences_grouped:
        for dna_sequence in dna_sequences:
            mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
            cumulative_coords = _get_cumulative_coords(mapped_coords)
            x, y = zip(*cumulative_coords)
            ax.plot(x, y, color=colors[count], label=labels[count])
        count += 1
    # only show unique labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()

