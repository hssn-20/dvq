from shiny import render
from shiny.express import input, ui
from datasets import load_dataset
import pandas as pd

### dvq stuff, obvs this will just be an import in the final version
from typing import Dict, Optional
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from tqdm import tqdm
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

ds = load_dataset('Hack90/virus_tiny')
df = pd.DataFrame(ds['train'])
virus = df['Organism_Name'].unique()
virus = {v: v for v in virus}

ui.input_selectize(  
    "selectize",  
    "Select options below:",
    virus,
    multiple=True,  
)  

@render.text
def value():
    return f"{input.selectize()}"

@render.plot
def plot():  
    ds = load_dataset('Hack90/virus_tiny')
    df = pd.DataFrame(ds['train'])
    df = df[df['Organism_Name'].isin(input.selectize())]
    # group by virus
    grouped = df.groupby('Organism_Name')['Sequence'].apply(list)
    # plot the comparison
    fig = plot_2d_comparison(grouped, grouped.index)
    return fig 