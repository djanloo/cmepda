"""Module for augmentation of data."""
import numpy as np
from rich.progress import track

from context import FeederProf
from context import constants


class Augment:
    def __init__(self, prof, N=10_000):
        self.data_indexes = prof.datum_indexes[-N:]
        self.dataset = np.empty(N, dtype=constants.funky_dtype)
        self.augmented_data = None
        self.start_number = prof.data_len
        self.directory = prof.folder
        self.N = N

        for j, idx in enumerate(self.data_indexes):
            fname = constants.FILENAME.format(name=idx)
            self.dataset[j] = np.load(f"{prof.folder}/{fname}")

    def augment_dataset(self):
        # Initialize a new record with the custom dtype
        new_record = np.empty(1, dtype=constants.funky_dtype)

        # keys of various types of augmentation
        aug_types = ['rot', 'flip_lr', 'flip_ud', 'flip_diag']

        # definitions
        index_record = self.start_number

        for record in track(self.dataset, total=self.N):
            # chiamare augment
            toa_dict = self.augment_matrix(record['toa'].squeeze())
            ts_list_of_dict = [self.augment_matrix(instant.reshape(9, 9)) for instant in record['time_series']]

            # ts_list_of_dict is a list of dictionaries, we want a dictionary of lists
            ts_dict_of_list = {k: [el[k] for el in ts_list_of_dict] for k in ts_list_of_dict[0]}  # 30L thx

            for key in aug_types:
                # assegnate new_record
                new_record['toa'] = np.array(toa_dict[key]).reshape((9, 9, 1))
                new_record['time_series'] = np.array(ts_dict_of_list[key]).reshape(80, 81)
                new_record['outcome'] = record['outcome']

                # Saving and updating index
                fname = constants.FILENAME.format(name=index_record)
                np.save(f'{self.directory}/{fname}', new_record)
                index_record += 1

    def augment_matrix(self, matrix):
        # Initialize a new record with the custom dtype
        new_rec_rot = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_lr = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_ud = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_diag = np.empty([9, 9], dtype=np.float32)

        # dictionary with all types of augment
        new_record = {'rot': new_rec_rot,
                      'flip_lr': new_rec_flip_lr,
                      'flip_ud': new_rec_flip_ud,
                      'flip_diag': new_rec_flip_diag}

        # arguments
        args = [90, 'left-right', 'upside-down', 'diagonal']

        for key, arg in zip(new_record.keys(), args):
            if 'rot' in key:
                new_record[key] = self.rotate_matrix(matrix, angle=arg)
            if 'flip' in key:
                new_record[key] = self.flip_matrix(matrix, flip_mode=arg)

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


if __name__ == '__main__':

    feeder_options = {
        "batch_size": 100,
        "input_fields": ["toa", "time_series"],
        "target_field": "outcome",
    }

    # Calling prof Albertino
    prof_train = FeederProf(
        "trained/albertino", constants.DIR_DATA_BY_ENTRY_AUG + "/test", **feeder_options, n_of_epochs=1
    )
    print(prof_train.data_len)

    # initialize and run augmentation
    aug = Augment(prof_train)
    aug.augment_dataset()
