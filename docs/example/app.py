from shiny import render
from shiny.express import input, ui
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import matplotlib
import numpy as np

############################################################# 2D Line Plot ########################################################
### dvq stuff, obvs this will just be an import in the final version
from typing import Dict, Optional
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from pathlib import Path
# using a faster style for plotting
mplstyle.use('fast')

# Mapping of nucleotides to float coordinates
mapping_easy = {
    'A': np.array([0.5, -0.8660254037844386]),
    'T': np.array([0.5, 0.8660254037844386]),
    'G': np.array([0.8660254037844386, -0.5]),
    'C': np.array([0.8660254037844386, 0.5]),
    'N': np.array([0, 0])
}

# coordinates for x+iy
Coord = namedtuple("Coord", ["x","y"])

# coordinates for a CGR encoding
CGRCoords = namedtuple("CGRCoords", ["N","x","y"])

# coordinates for each nucleotide in the 2d-plane
DEFAULT_COORDS = dict(A=Coord(1,1),C=Coord(-1,1),G=Coord(-1,-1),T=Coord(1,-1))

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
    return fig

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
    return fig


############################################################# Virus Dataset ########################################################
ds = load_dataset('Hack90/virus_tiny')
df = pd.DataFrame(ds['train'])
virus = df['Organism_Name'].unique()
virus = {v: v for v in virus}

############################################################# Filter and Select ########################################################
def filter_and_select(group):
    if len(group) >= 3:
        return group.head(3)
    
############################################################# Wens Method ########################################################
import numpy as np

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

def moment_of_inertia(sequence, WEIGHTS, L=5000, E=0.0375):
    subsequences = _get_subsequences(sequence)
    polar_coordinates = {subsequence: _calculate_coordinates_fixed(indices, len(sequence)) for subsequence, indices in subsequences.items()}
    weightings = _calculate_weighting_full(sequence, WEIGHTS, L=L, E=E)
    return _moments_of_inertia(polar_coordinates, weightings)


def similarity_wen(sequence1, sequence2, WEIGHTS, L=5000, E=0.0375):
    L = min(len(sequence1), len(sequence2))
    inertia1 = moment_of_inertia(sequence1, WEIGHTS, L=L, E=E)
    inertia2 = moment_of_inertia(sequence2, WEIGHTS, L=L, E=E)
    similarity = np.sqrt(sum((x - y)**2 for x, y in zip(inertia1, inertia2)))
    return similarity
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def wens_method_heatmap(df, virus_species):
    # Create a dataframe to store the similarity values
    similarity_df = pd.DataFrame(index=virus_species, columns=virus_species)
    # Fill the dataframe with similarity values
    for virus1 in virus_species:
        for virus2 in virus_species:
            if virus1 == virus2:
                similarity_df.loc[virus1, virus2] = 0
            else:
                sequence1 = df[df['Organism_Name'] == virus1]['Sequence'].values[0]
                sequence2 = df[df['Organism_Name'] == virus2]['Sequence'].values[0]
                similarity = similarity_wen(sequence1, sequence2, WEIGHTS)
                similarity_df.loc[virus1, virus2] = similarity
    similarity_df = similarity_df.apply(pd.to_numeric)

    # Optional: Handle NaN values if your similarity computation might result in them
    # similarity_df.fillna(0, inplace=True)

    fig, ax = plt.subplots()
    # Plotting
    im = ax.imshow(similarity_df, cmap="YlGn")
    ax.set_xticks(np.arange(len(virus_species)), labels=virus_species)
    ax.set_yticks(np.arange(len(virus_species)), labels=virus_species)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom")

    
    return fig

    
############################################################# ColorSquare ########################################################
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

def _fill_spiral(matrix, seq_colors, k):
        left, top, right, bottom = 0, 0, k-1, k-1
        index = 0
        while left <= right and top <= bottom:
            for i in range(left, right + 1):  # Top row
                if index < len(seq_colors):
                    matrix[top][i] = seq_colors[index]
                    index += 1
            top += 1
            for i in range(top, bottom + 1):  # Right column
                if index < len(seq_colors):
                    matrix[i][right] = seq_colors[index]
                    index += 1
            right -= 1
            for i in range(right, left - 1, -1):  # Bottom row
                if index < len(seq_colors):
                    matrix[bottom][i] = seq_colors[index]
                    index += 1
            bottom -= 1
            for i in range(bottom, top - 1, -1):  # Left column
                if index < len(seq_colors):
                    matrix[i][left] = seq_colors[index]
                    index += 1
            left += 1


def _generate_color_square(sequence,virus, save=False, count=0, label=None):
    # Define the sequence and corresponding colors with indices
    colors = {'a': 0, 't': 1, 'c': 2, 'g': 3, 'n': 4}  # Assign indices to each color
    seq_colors = [colors[char] for char in sequence.lower()]  # Map the sequence to color indices

    # Calculate k (size of the square)
    k = math.ceil(math.sqrt(len(sequence)))

    # Initialize a k x k matrix filled with the index for 'white'
    matrix = np.full((k, k), colors['n'], dtype=int)

    # Fill the matrix in a clockwise spiral
    _fill_spiral(matrix, seq_colors, k)

    # Define a custom color map for plotting
    cmap = ListedColormap(['red', 'green', 'yellow', 'blue', 'white'])

    # Plot the matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    if label:
        plt.title(label)
    plt.axis('off')  # Hide the axes
    if save:
        plt.savefig(f'color_square_{virus}_{count}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_color_square(df, virus_species):
    ncols = 3 
    nrows = len(virus_species)
    fig, axeses = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
    )
    for i in range(0, ncols * nrows):
        row = i // ncols
        col = i % ncols
        axes = axeses[row, col]
        data = df[i]
        virus = virus_species[row]
                # Define the sequence and corresponding colors with indices
        colors = {'a': 0, 't': 1, 'c': 2, 'g': 3, 'n': 4}  # Assign indices to each color
        seq_colors = [colors[char] for char in data.lower()]  # Map the sequence to color indices

        # Calculate k (size of the square)
        k = math.ceil(math.sqrt(len(data)))

        # Initialize a k x k matrix filled with the index for 'white'
        matrix = np.full((k, k), colors['n'], dtype=int)

        # Fill the matrix in a clockwise spiral
        _fill_spiral(matrix, seq_colors, k)

        # Define a custom color map for plotting
        cmap = ListedColormap(['red', 'green', 'yellow', 'blue', 'white'])
        axes.imshow(matrix, cmap=cmap, interpolation='nearest')
        axes.set_title(virus)
    return fig
    
    

def generate_color_square(sequence,virus, multi=False, save=False, label=None):
    if multi:
        for i,seq in enumerate(sequence):
            _generate_color_square(seq, virus,save, i, label[i] if label else None)
    else:
        _generate_color_square(sequence, save, label=label)


############################################################# FCGR ########################################################

from typing import Dict, Optional
from collections import namedtuple

# coordinates for x+iy
Coord = namedtuple("Coord", ["x","y"])

# coordinates for a CGR encoding
CGRCoords = namedtuple("CGRCoords", ["N","x","y"])

# coordinates for each nucleotide in the 2d-plane
DEFAULT_COORDS = dict(A=Coord(1,1),C=Coord(-1,1),G=Coord(-1,-1),T=Coord(1,-1))

class CGR:
    "Chaos Game Representation for DNA"
    def __init__(self, coords: Optional[Dict[chr,tuple]]=None):
        self.nucleotide_coords = DEFAULT_COORDS if coords is None else coords
        self.cgr_coords = CGRCoords(0,0,0)

    def nucleotide_by_coords(self,x,y):
        "Get nucleotide by coordinates (x,y)"
        # filter nucleotide by coordinates
        filtered = dict(filter(lambda item: item[1] == Coord(x,y), self.nucleotide_coords.items()))

        return list(filtered.keys())[0]

    def forward(self, nucleotide: str):
        "Compute next CGR coordinates"
        x = (self.cgr_coords.x + self.nucleotide_coords.get(nucleotide).x)/2
        y = (self.cgr_coords.y + self.nucleotide_coords.get(nucleotide).y)/2

        # update cgr_coords
        self.cgr_coords = CGRCoords(self.cgr_coords.N+1,x,y)

    def backward(self,):
        "Compute last CGR coordinates. Current nucleotide can be inferred from (x,y)"
        # get current nucleotide based on coordinates
        n_x,n_y = self.coords_current_nucleotide()
        nucleotide = self.nucleotide_by_coords(n_x,n_y)

        # update coordinates to the previous one
        x = 2*self.cgr_coords.x - n_x
        y = 2*self.cgr_coords.y - n_y

        # update cgr_coords
        self.cgr_coords = CGRCoords(self.cgr_coords.N-1,x,y)

        return nucleotide

    def coords_current_nucleotide(self,):
        x = 1 if self.cgr_coords.x>0 else -1
        y = 1 if self.cgr_coords.y>0 else -1
        return x,y

    def encode(self, sequence: str):
        "From DNA sequence to CGR"
        # reset starting position to (0,0,0)
        self.reset_coords()
        for nucleotide in sequence:
            self.forward(nucleotide)
        return self.cgr_coords

    def reset_coords(self,):
        self.cgr_coords = CGRCoords(0,0,0)

    def decode(self, N:int, x:int, y:int)->str:
        "From CGR to DNA sequence"
        self.cgr_coords = CGRCoords(N,x,y)

        # decoded sequence
        sequence = []

        # Recover the entire genome
        while self.cgr_coords.N>0:
            nucleotide = self.backward()
            sequence.append(nucleotide)
        return "".join(sequence[::-1])
    
    
from itertools import product
from collections import defaultdict
import numpy as np

class FCGR(CGR):
    """Frequency matrix CGR
    an (2**k x 2**k) 2D representation will be created for a
    n-long sequence.
    - k represents the k-mer.
    - 2**k x 2**k = 4**k the total number of k-mers (sequences of length k)
    - pixel value correspond to the value of the frequency for each k-mer
    """

    def __init__(self, k: int,):
        super().__init__()
        self.k = k # k-mer representation
        self.kmers = list("".join(kmer) for kmer in product("ACGT", repeat=self.k))
        self.kmer2pixel = self.kmer2pixel_position()

    def __call__(self, sequence: str):
        "Given a DNA sequence, returns an array with his frequencies in the same order as FCGR"
        self.count_kmers(sequence)

        # Create an empty array to save the FCGR values
        array_size = int(2**self.k)
        freq_matrix = np.zeros((array_size,array_size))

        # Assign frequency to each box in the matrix
        for kmer, freq in self.freq_kmer.items():
            pos_x, pos_y = self.kmer2pixel[kmer]
            freq_matrix[int(pos_x)-1,int(pos_y)-1] = freq
        return freq_matrix

    def count_kmer(self, kmer):
        if "N" not in kmer:
            self.freq_kmer[kmer] += 1

    def count_kmers(self, sequence: str):
        self.freq_kmer = defaultdict(int)
        # representativity of kmers
        last_j = len(sequence) - self.k + 1
        kmers  = (sequence[i:(i+self.k)] for i in range(last_j))
        # count kmers in a dictionary
        list(self.count_kmer(kmer) for kmer in kmers)

    def kmer_probabilities(self, sequence: str):
        self.probabilities = defaultdict(float)
        N=len(sequence)
        for key, value in self.freq_kmer.items():
            self.probabilities[key] = float(value) / (N - self.k + 1)

    def pixel_position(self, kmer: str):
        "Get pixel position in the FCGR matrix for a k-mer"

        coords = self.encode(kmer)
        N,x,y = coords.N, coords.x, coords.y

        # Coordinates from [-1,1]² to [1,2**k]²
        np_coords = np.array([(x + 1)/2, (y + 1)/2]) # move coordinates from [-1,1]² to [0,1]²
        np_coords *= 2**self.k # rescale coordinates from [0,1]² to [0,2**k]²
        x,y = np.ceil(np_coords) # round to upper integer

        # Turn coordinates (cx,cy) into pixel (px,py) position
        # px = 2**k-cy+1, py = cx
        return 2**self.k-int(y)+1, int(x)

    def kmer2pixel_position(self,):
        kmer2pixel = dict()
        for kmer in self.kmers:
            kmer2pixel[kmer] = self.pixel_position(kmer)
        return kmer2pixel
    

from tqdm import tqdm
from pathlib import Path

import numpy as np


class GenerateFCGR:
    def __init__(self,  kmer: int = 5, ):
        self.kmer = kmer
        self.fcgr = FCGR(kmer)
        self.counter = 0 # count number of time a sequence is converted to fcgr


    def __call__(self, list_fasta,):

        for fasta in tqdm(list_fasta, desc="Generating FCGR"):
            self.from_fasta(fasta)




    def from_seq(self, seq: str):
        "Get FCGR from a sequence"
        seq = self.preprocessing(seq)
        chaos = self.fcgr(seq)
        self.counter +=1
        return chaos

    def reset_counter(self,):
        self.counter=0

    @staticmethod
    def preprocessing(seq):
        seq = seq.upper()
        for letter in seq:
          if letter not in "ATCG":
            seq = seq.replace(letter,"N")
        return seq

def plot_fcgr(df, virus_species):
    ncols = 3 
    nrows = len(virus_species)
    fig, axeses = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
    )
    for i in range(0, ncols * nrows):
        row = i // ncols
        col = i % ncols
        axes = axeses[row, col]
        data = df[i].upper()
        chaos = GenerateFCGR().from_seq(seq=data)
        virus = virus_species[row]
        axes.imshow(chaos)
        axes.set_title(virus)
    return fig

############################################################# Persistant Homology ########################################################
import numpy as np
import persim
import ripser
import matplotlib.pyplot as plt

NUCLEOTIDE_MAPPING = {
    'a': np.array([1, 0, 0, 0]),
    'c': np.array([0, 1, 0, 0]),
    'g': np.array([0, 0, 1, 0]),
    't': np.array([0, 0, 0, 1])
}

def encode_nucleotide_to_vector(nucleotide):
    return NUCLEOTIDE_MAPPING.get(nucleotide)

def chaos_4d_representation(dna_sequence):
    points = [encode_nucleotide_to_vector(dna_sequence[0])]
    for nucleotide in dna_sequence[1:]:
        vector = encode_nucleotide_to_vector(nucleotide)
        if 
        next_point = 0.5 * (points[-1] + vector)
        points.append(next_point)
    return np.array(points)

def persistence_homology(dna_sequence, multi=False, plot=False, sample_rate=7):
    if multi:
        c4dr_points = np.array([chaos_4d_representation(sequence) for sequence in dna_sequence])
        dgm_dna = [ripser.ripser(points[::sample_rate], maxdim=1)['dgms'] for points in c4dr_points]
        if plot:
            persim.plot_diagrams([dgm[1] for dgm in dgm_dna], labels=[f'sequence {i}' for i in range(len(dna_sequence))])
    else:
        c4dr_points = chaos_4d_representation(dna_sequence)
        dgm_dna = ripser.ripser(c4dr_points[::sample_rate], maxdim=1)['dgms']
        if plot:
            persim.plot_diagrams(dgm_dna[1])
    return dgm_dna

def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None
):
    """A helper function to plot persistence diagrams. 

    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with

        .. code:: python

            import matplotlib as mpl
            print(mpl.styles.available)

    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    fig, ax = plt.subplots() if ax is None else ax
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()
    return fig, ax


def plot_persistence_homology(df, virus_species):
    if len(virus_species.unique()) > 1:
        c4dr_points = [chaos_4d_representation(sequence.lower()) for sequence in df]
        dgm_dna = [ripser.ripser(points[::10], maxdim=1)['dgms'] for points in c4dr_points]
        labels =[f'{virus_specie}_{i}' for i, virus_specie in enumerate(virus_species)]
        fig, ax = plot_diagrams([dgm[1] for dgm in dgm_dna], labels=virus_species)
    else:
        c4dr_points = [chaos_4d_representation(sequence.lower()) for sequence in df]
        dgm_dna = [ripser.ripser(points[::10], maxdim=1)['dgms'] for points in c4dr_points]
        labels =[f'{virus_specie}_{i}' for i, virus_specie in enumerate(virus_species)]
        fig, ax = plot_diagrams([dgm[1] for dgm in dgm_dna], labels=labels)
    return fig
    
def compare_persistence_homology(dna_sequence1, dna_sequence2):
    dgm_dna1 = persistence_homology(dna_sequence1)
    dgm_dna2 = persistence_homology(dna_sequence2)
    distance = persim.sliced_wasserstein(dgm_dna1[1], dgm_dna2[1])
    return distance

############################################################# UI #################################################################
ui.page_opts(fillable=True)
ui.panel_title("Whats in your DNA?")
with ui.layout_columns():
    with ui.card():
        ui.input_selectize(  
            "virus_selector",  
            "Select viruses:",
            virus,
            multiple=True,  
        )  
    with ui.card():
        ui.input_selectize(  
        "plot_type",  
        "Select method:",
        ["Chaos Game Representation", "2D Line", "ColorSquare", "Persistant Homology", "Wens Method"],
        multiple=False,  
    )
        
############################################################# Plotting ########################################################
here = Path(__file__).parent
@render.plot
def plot():  
    ds = load_dataset('Hack90/virus_tiny')
    df = pd.DataFrame(ds['train'])
    df = df[df['Organism_Name'].isin(input.virus_selector())]
    # group by virus
    grouped = df.groupby('Organism_Name')['Sequence'].apply(list)
    # plot the comparison
    fig = None
    if input.plot_type() == "2D Line":
        fig = plot_2d_comparison(grouped, grouped.index)
    if input.plot_type() == "ColorSquare":
        filtered_df = df.groupby('Organism_Name').apply(filter_and_select).reset_index(drop=True)
        fig = plot_color_square(filtered_df['Sequence'], filtered_df['Organism_Name'].unique())
    if input.plot_type() == "Wens Method":
        fig = wens_method_heatmap(df, df['Organism_Name'].unique())
    if input.plot_type() == "Chaos Game Representation":
        filtered_df = df.groupby('Organism_Name').apply(filter_and_select).reset_index(drop=True)
        fig = plot_fcgr(filtered_df['Sequence'], df['Organism_Name'].unique())
    if input.plot_type() == "Persistant Homology":
        filtered_df = df.groupby('Organism_Name').apply(filter_and_select).reset_index(drop=True)
        fig = plot_persistence_homology(filtered_df['Sequence'], filtered_df['Organism_Name'])
    return fig

# @render.image  
# def image():
#     img = None
#     if input.plot_type() == "ColorSquare":
#         img = {"src": f"color_square_{input.virus_selector()[0]}_0.png", "alt": "ColorSquare"} 
#         return img 
#     return img 
