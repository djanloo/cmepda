"""Module for augmentation of data."""
import numpy as np

def flip_matrix(matrix, mode):
    """Flips the matrix with different axis modes.
    Modes can be 'up-down', 'left-right', 'diagonal'."""
    if mode == "up-down":
        matrix = matrix[::-1]
        return matrix

    if mode == 'left-right':
        for row in range(len(matrix)):
            matrix[row] = matrix[row][::-1]
            return matrix

    if mode == 'diagonal':
        matrix = np.transpose(matrix)
        return matrix

def rotate_matrix(matrix, pi_multiple=0.5):
    """Rotate the matrix of different multiples of pi."""