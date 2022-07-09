"""Module for augmentation of data."""
import numpy as np
import constants


class Augument:
    def __init__(self, dataset):
        self.dataset = dataset
        self.augmented_data = None

    def augment_dataset(self, dataset, tot):
        # Initialize a new record with the custom dtype
        new_rec_rot = np.empty(1, dtype=constants.funky_dtype)
        new_rec_flip_lr = np.empty(1, dtype=constants.funky_dtype)
        new_rec_flip_ud = np.empty(1, dtype=constants.funky_dtype)
        new_rec_flip_diag = np.empty(1, dtype=constants.funky_dtype)

        for i, file in enumerate(dataset):
            # rotations
            tot += 1
            new_rec_rot['toa'] = self.rotate_matrix(dataset['toa'])
            new_rec_rot['time_series'] = self.rotate_matrix(dataset['time_series'])
            new_rec_rot['outcome'] = dataset["outcome"]
            np.save(f'{constants.DIR_DATA_BY_ENTRY}/part_{tot}.npy', new_rec_rot)

            # flips left-right
            tot += 1
            new_rec_flip_lr['toa'] = self.flip_matrix(dataset['toa'], mode='left-right')
            new_rec_flip_lr['time_series'] = self.flip_matrix(dataset['time_series'], mode='left-right')
            new_rec_flip_lr['outcome'] = dataset["outcome"]
            np.save(f'part_{tot}.npy', new_rec_flip_lr)

            # flips up-down
            tot += 1
            new_rec_flip_ud['toa'] = self.flip_matrix(dataset['toa'], mode='up-down')
            new_rec_flip_ud['time_series'] = self.flip_matrix(dataset['time_series'], mode='up-down')
            new_rec_flip_ud['outcome'] = dataset["outcome"]
            np.save(f'part_{tot}.npy', new_rec_flip_ud)

            # flip diagonal
            tot += 1
            new_rec_flip_diag['toa'] = self.flip_matrix(dataset['toa'], mode='diagonal')
            new_rec_flip_diag['time_series'] = self.flip_matrix(dataset['time_series'], mode='diagonal')
            new_rec_flip_diag['outcome'] = dataset["outcome"]
            np.save(f'part_{tot}.npy', new_rec_flip_diag)

            # new records
            new_records = [new_rec_rot, new_rec_flip_lr, new_rec_flip_ud, new_rec_flip_diag]
            new_records = {'rot': [new_rec_rot, self.rotate_matrix()],
                           'flip_lr': [new_rec_flip_lr, ]}

            for n_rec in zip(new_records):
                # Initialize a new record with the custom dtype
                n_rec = np.empty(1, dtype=constants.funky_dtype)

                # augmenting with all the operations

    def augment_matrix(self, matrix):
        # Initialize a new record with the custom dtype
        new_rec_rot = np.empty(1, dtype=constants.funky_dtype)
        new_rec_flip_lr = np.empty(1, dtype=constants.funky_dtype)
        new_rec_flip_ud = np.empty(1, dtype=constants.funky_dtype)
        new_rec_flip_diag = np.empty(1, dtype=constants.funky_dtype)

        new_records = {'rot': [new_rec_rot, self.rotate_matrix()],
                       'flip_lr': [new_rec_flip_lr, self.flip_matrix(matrix, mode='lr')]}

    @staticmethod
    def rotate_matrix_set(angle=90):
        """Rotate the matrix of different angles."""
        rot_angle = angle

        def rotate(matrix):
            if angle == 90:
                matrix = np.rot90(matrix)
                return matrix

            # TODO:evaluate if other rotations are needed


    @staticmethod
    def flip_matrix_set(mode):
        """Flips the matrix with different axis modes.
        Modes can be 'up-down', 'left-right', 'diagonal'."""
        flip_mode = mode

        def flip(matrix):
            if flip_mode == "up-down":
                matrix = np.flipud(matrix)
                return matrix

            if flip_mode == 'left-right':
                matrix = np.fliplr(matrix)
                return matrix

            if flip_mode == 'diagonal':
                matrix = np.transpose(matrix)
                return matrix
