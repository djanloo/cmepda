"""Module for augmentation of data."""
import numpy as np
from cloudatlas import constants


class Augment:
    def __init__(self, dataset):
        self.dataset = dataset
        self.augmented_data = None

    def augment_dataset(self, dataset, tot_files):
        # Initialize a new record with the custom dtype
        new_record = np.empty(1, dtype=constants.funky_dtype)

        # keys of various types of augmentation
        aug_types = ['rot', 'flip_lr', 'flip_ud', 'flip_diag']

        # definitions
        index_record = tot_files + 1

        for file in dataset:
            for key in aug_types:
                # calling function of augmentation
                new_record['toa'] = self.augment_matrix(file['toa'])[key]
                new_record['time_series'] = np.array(
                    [self.augment_matrix(_)[key] for _ in file['time_series'].reshape(-1, 9, 9)])
                new_record['output'] = file['output']

                # Saving and updating index
                np.save(f'{constants.DIR_DATA_BY_ENTRY}/part_{index_record}.npy')
                index_record += 1

    def augment_matrix(self, matrix):
        # Initialize a new record with the custom dtype
        new_rec_rot = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_lr = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_ud = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_diag = np.empty([9, 9], dtype=np.float32)

        # dictionary with all types of augment
        new_record = {'rot': [new_rec_rot, 90],
                      'flip_lr': [new_rec_flip_lr, 'left-right'],
                      'flip_ud': [new_rec_flip_ud, 'upside-down'],
                      'flip_diag': [new_rec_flip_diag, 'diagonal']}

        for key in new_record.keys():
            if 'rot' in key:
                new_record[key][0] = self.rotate_matrix(matrix, angle=new_record[key][1])
            if 'flip' in key:
                new_record[key][0] = self.flip_matrix(matrix, flip_mode=new_record[key][1])

        return new_record

    @staticmethod
    def rotate_matrix(matrix, angle=90):
        """Rotate the matrix of different angles."""
        if angle == 90:
            matrix = np.rot90(matrix)
            return matrix

        # TODO:evaluate if other rotations are needed

    @staticmethod
    def flip_matrix(matrix, flip_mode):
        """Flips the matrix with different axis modes.
        Modes can be 'upside-down', 'left-right', 'diagonal'."""

        # Check on the flip mode
        if flip_mode == "upside-down":
            matrix = np.flipud(matrix)
            return matrix

        if flip_mode == 'left-right':
            matrix = np.fliplr(matrix)
            return matrix

        if flip_mode == 'diagonal':
            matrix = np.transpose(matrix)
            return matrix
