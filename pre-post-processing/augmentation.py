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
        new_record_rot = np.empty(1, dtype=funky_dtype)
        new_record_flip_lr = np.empty(1, dtype=funky_dtype)
        new_record_flip_ud = np.empty(1, dtype=funky_dtype)
        new_record_flip_diag = np.empty(1, dtype=funky_dtype)

        for i, file in enumerate(dataset):
            # rotations
            tot += 1
            new_record_rot['toa'] = self.rotate_matrix(dataset['toa'])
            new_record_rot['time_series'] = self.rotate_matrix(dataset['time_series'])
            new_record_rot['outcome'] = dataset["outcome"]
            np.save(f'part_{tot}.npy', new_record_rot)

            # flips left-right
            tot += 1
            new_record_flip_lr['toa'] = self.flip_matrix(dataset['toa'], mode='left-right')
            new_record_flip_lr['time_series'] = self.flip_matrix(dataset['time_series'], mode='left-right')
            new_record_flip_lr['outcome'] = dataset["outcome"]
            np.save(f'part_{tot}.npy', new_record_flip_lr)

            # flips up-down
            tot += 1
            new_record_flip_ud['toa'] = self.flip_matrix(dataset['toa'], mode='up-down')
            new_record_flip_ud['time_series'] = self.flip_matrix(dataset['time_series'], mode='up-down')
            new_record_flip_ud['outcome'] = dataset["outcome"]
            np.save(f'part_{tot}.npy', new_record_flip_ud)

            # flip diagonal
            tot += 1
            new_record_flip_diag['toa'] = self.flip_matrix(dataset['toa'], mode='diagonal')
            new_record_flip_diag['time_series'] = self.flip_matrix(dataset['time_series'], mode='diagonal')
            new_record_flip_diag['outcome'] = dataset["outcome"]
            np.save(f'part_{tot}.npy', new_record_flip_diag)


    @staticmethod
    def rotate_matrix(matrix, angle=90):
        """Rotate the matrix of different angles."""
        if angle == 90:
            matrix = np.flipud(matrix)
            matrix = np.transpose(matrix)
            return matrix

    @staticmethod
    def flip_matrix(matrix, mode):
        """Flips the matrix with different axis modes.
        Modes can be 'up-down', 'left-right', 'diagonal'."""
        if mode == "up-down":
            matrix = np.flipud(matrix)
            return matrix

        if mode == 'left-right':
            for row in range(len(matrix)):
                matrix[row] = np.fliplr(matrix)
                return matrix

        if mode == 'diagonal':
            matrix = np.transpose(matrix)
            return matrix
