"""Module for augmentation of data."""
import numpy as np

funky_dtype = np.dtype(
    [
        ("outcome", np.float64),
        ("time_series", np.float32, (80, 81)),
        ("toa", np.float32, (9, 9, 1)),
    ]
)

class Augument:
    def __init__(self, dataset):
        self.dataset = dataset
        self.augmented_data = None

    def augment_dataset(self, dataset, tot):
        # Initialize a new record with the custom dtype
        new_record = np.empty(1, dtype=funky_dtype)

        for i, file in enumerate(dataset):
            new_record['toa'] = self.rotate_matrix(dataset['toa'])
            new_record['time']


    @staticmethod
    def rotate_matrix(matrix, angle=90):
        """Rotate the matrix of different angles."""
        if angle == 90:
            for column in range(len(matrix)):
                matrix[:, column] = matrix[:, column][::-1]
                matrix = np.transpose(matrix)

    @staticmethod
    def flip_matrix(matrix, mode):
        """Flips the matrix with different axis modes.
        Modes can be 'up-down', 'left-right', 'diagonal'."""
        if mode == "up-down":
            matrix = np.flipud(matrix)
            return matrix

        if mode == 'left-right':
            for row in range(len(matrix)):
                matrix[row] =
                return matrix

        if mode == 'diagonal':
            matrix = np.transpose(matrix)
            return matrix
