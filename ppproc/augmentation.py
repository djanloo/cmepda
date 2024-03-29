"""Module for augmentation of data."""
import os

import numpy as np
from rich.progress import track

from context import FeederProf
from context import constants



class Augment:
    """Class for executing data augmentation.
    
    One can choose if augment the more difficult data to predict (given back by the class FeederProf) or data over
    a certain height threshold. 'prof' and 'height_threshold' can't be both different from None.
    
    Args:
        dataset_dir (:path): path to the dataset to augment, if left None a prof should be given and will automatically find one
        prof (:obj:'FeederProf'): instance of FeederProf class who gives back data sorted by difficulty.
        N (int, optional): number of data to augment
        height_threshold (int): if given sets the threshold for augmenting data over that height, at a maximum of N data.
        """
    def __init__(self, dataset_dir=None, prof=None, N=15_000, height_threshold=None, test_mode=False):
        self.directory = dataset_dir
        self.start_number = len(os.listdir(self.directory))

        # if one have chosen to augment more difficult data
        if prof is not None: 
            self.data_indexes = prof.datum_indexes[-N:]  # takes data-indexes of data sorted by difficulty
            self.start_number = prof.data_len  #
            self.directory = prof.folder

        # check if there's a directory
        if self.directory is None and test_mode is False:
            raise NotImplementedError("A directory or a prof should be given!")

        self.dataset = np.empty(N, dtype=constants.funky_dtype) # initialize an empty dataset of len N with right dtype
        self.augmented_data = None
        self.N = N

        # useful stuff
        dummy = np.empty(0, dtype=constants.funky_dtype)
        count_right_files = 0

        # augmentation by height
        if height_threshold is not None:
            #assing height threshold
            self.height_threshold = height_threshold

            # load dataset
            for index, fname in enumerate(os.listdir(self.directory)):
                dummy = np.load(f"{self.directory}/{fname}")  # open one data at once and put it into dummy
                if dummy['outcome'] > self.height_threshold:  # check on the height value
                    self.dataset[count_right_files] = dummy  # put into dataset
                    count_right_files += 1
                    if count_right_files >= self.N:  # breaks when we've got the right number of samples
                        break
                else:
                    continue

        # check if there are lines left empty in self.dataset, if so remove them
        if count_right_files < self.N:
            lines_to_remove = self.N - count_right_files
            self.dataset = np.delete(self.dataset, np.s_[lines_to_remove-1:-1])
            
        # augmentation by difficulty
        if prof is not None:
            for j, idx in enumerate(self.data_indexes):
                fname = constants.FILENAME.format(name=idx)
                self.dataset[j] = np.load(f"{self.directory}/{fname}")

        # check if both are not None
        if prof is not None and height_threshold is not None and test_mode is False:
            raise NotImplementedError("prof and height_threshold can't be both different from None! "
                                      "If running in test mode put test_mode=True")

    def augment_dataset(self):
        """Effectively realize the augmentation on the initialized dataset.
        Augmentation consists in rotation by 90°, horizontal flip, vertical flip, diagonal flip."""
        # Initialize a new record with the custom dtype
        new_record = np.empty(1, dtype=constants.funky_dtype)

        # keys of various types of augmentation
        aug_types = ["rot", "flip_lr", "flip_ud", "flip_diag"]

        # definition
        # Files in folder are [0, .., len(files) - 1] (extrema included)
        # so it should start saving files from index = len(files)
        index_record = self.start_number

        for record in track(self.dataset, total=len(self.dataset)):

            # check on the height threshold
            if self.height_threshold:
                if record["outcome"] < self.height_threshold:
                    continue

            # chiamare augment
            toa_dict = self.augment_matrix(record["toa"].squeeze())
            ts_list_of_dict = [
                self.augment_matrix(instant.reshape(9, 9))
                for instant in record["time_series"]
            ]

            # ts_list_of_dict is a list of dictionaries, we want a dictionary of lists
            ts_dict_of_list = {
                k: [el[k] for el in ts_list_of_dict] for k in ts_list_of_dict[0]
            }

            for key in aug_types:
                # assegnate new_record
                new_record["toa"] = np.array(toa_dict[key]).reshape((9, 9, 1))
                new_record["time_series"] = np.array(ts_dict_of_list[key]).reshape(
                    80, 81
                )
                new_record["outcome"] = record["outcome"]

                # Saving and updating index
                fname = constants.FILENAME.format(name=index_record)
                np.save(f"{self.directory}/{fname}", new_record)

                index_record += 1

    def augment_matrix(self, matrix):
        """Augmentation of a single matrix."""
        # Initialize a new record with the custom dtype
        new_rec_rot = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_lr = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_ud = np.empty([9, 9], dtype=np.float32)
        new_rec_flip_diag = np.empty([9, 9], dtype=np.float32)

        # dictionary with all types of augment
        new_record = {
            "rot": new_rec_rot,
            "flip_lr": new_rec_flip_lr,
            "flip_ud": new_rec_flip_ud,
            "flip_diag": new_rec_flip_diag,
        }

        # arguments
        args = [90, "left-right", "upside-down", "diagonal"]

        for key, arg in zip(new_record.keys(), args):
            if "rot" in key:
                new_record[key] = self.rotate_matrix(matrix, angle=arg)
            if "flip" in key:
                new_record[key] = self.flip_matrix(matrix, flip_mode=arg)

        return new_record

    @staticmethod
    def rotate_matrix(matrix, angle=90):
        """Rotate the matrix by 90°."""
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

        if flip_mode == "left-right":
            matrix = np.fliplr(matrix)
            return matrix

        if flip_mode == "diagonal":
            matrix = np.transpose(matrix)
            return matrix


if __name__ == "__main__":

    # initialize and run augmentation
    aug = Augment(dataset_dir="data_by_entry_height/validation", height_threshold=850)
    aug.augment_dataset()
