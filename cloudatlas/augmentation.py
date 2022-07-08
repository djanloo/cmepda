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


def rotate_matrix(matrix, angle=90):
    """Rotate the matrix of different angles."""
    if angle == 90:
        for column in range(len(matrix)):
            matrix[:, column] = matrix[:, column][::-1]
            matrix = np.transpose(matrix)
