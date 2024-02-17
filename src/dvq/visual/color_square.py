import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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


def _generate_color_square(sequence, save=False, count=0):
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
    plt.axis('off')  # Hide the axes
    if save:
        plt.savefig(f'color_square_{count}.png', dpi=300, bbox_inches='tight')
    plt.show()    
               
               
def generate_color_square(sequence, multi=False, save=False):
    if multi:
        for i,seq in enumerate(sequence):
            _generate_color_square(seq, save, i)
    else:
        _generate_color_square(sequence, save)
    
