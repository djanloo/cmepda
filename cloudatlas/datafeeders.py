"""Module for data feeders"""
import numpy as np
import os
from os import listdir
from os.path import isfile, join, exists
from rich.progress import track
from rich import print
import keras
from keras.models import load_model

# Test
from matplotlib import pyplot as plt
import utils

# Turn off keras warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROF_SAVEFILE = "prof_knowledge.npy"  # Must have format


class DataFeeder(keras.utils.Sequence):
    """Data generator that uses a `single-file` splitted dataset of numpy structured arrays.

    This means that the single record must have fields. Multiple inputs are supported.

    Args:
        folder (:obj:`str`): the folder where the dataset is stored.
        input_fields (:obj:`list` or :obj:`str`): the list of strings that specify the field names.
        target_field (:obj:`str`): the name of the target field.
        batch_size (int, optional): the batch size. Default is 32.
        shuffle (bool, optional): enable shuffling dataset on_epoch_end
    """

    def __init__(
        self, folder, input_fields=None, target_field=None, batch_size=32, shuffle=True
    ):
        ## TODO: default (in, tar) if no fields are provided
        if input_fields is None or target_field is None:
            raise NotImplementedError("input_fields and target_field must be given.")

        self.folder = folder
        self.batch_size = batch_size
        self.shuffle = shuffle

        # The fields of the array that will be feeded into the net as (in, target)
        self.input_fields = input_fields
        self.target_field = target_field

        # Checks for multiple inputs
        self.multiple_inputs = hasattr(input_fields, "__iter__")

        # Loads files names' preventing to load subfolders
        self.files = [
            file
            for file in os.listdir(self.folder)
            if os.path.isfile(join(self.folder, file))
        ]
        ## WARNING: files are not in order, even if ``sorted()`` is applied
        # This should not be a problem since indexes and files are one to one

        self.data_len = len(self.files)
        print(
            f"Found { self.data_len} files in [red]{self.folder}[/red]: {[self.files[i] for i in [1,2,3]]}.."
        )

        # Gets the dtype of the saved data form first entry
        self.datum_dtype = np.load(f"{self.folder}/part_0.npy").dtype

        # Data must be indexed by continuous integers
        self.datum_indexes = np.arange(self.data_len)
        # Shuffles
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch"""
        return int(np.floor(self.data_len / self.batch_size))

    def __getitem__(self, batch_index):
        """Gives one batch of data"""
        # print(f"Using DataFeederKeras __getitem__()")
        # Gives the daum indexes for the batch_index block in the order specified by the shuffle
        indexes = self.datum_indexes[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]

        # Generate data
        net_input, net_target = self.data_generation(indexes)

        # Test for curriculum learning: save the indexes of the batch
        self.last_batch_indexes = np.array(indexes)

        return net_input, net_target

    def on_epoch_end(self):
        """Shuffles indexes after each epoch"""
        self.datum_indexes = np.arange(self.data_len)
        if self.shuffle:
            print(f"[blue]Shuffled indexes[/blue] in DataFeederKeras({self.folder})")
            np.random.shuffle(self.datum_indexes)

    def data_generation(self, batch_datum_indexes):
        """Loads data and returns a batch"""
        # Return format must be ([array_input1, array_input2], array_of_targets)
        # Not array((in, tar))
        # Neither array([[in1, in2],
        #                [in1, in2]],
        #                [t1, t2])
        batch_rows = np.empty(self.batch_size, dtype=self.datum_dtype)
        for row, datum_index in enumerate(batch_datum_indexes):
            batch_rows[row] = np.load(f"{self.folder}/part_{datum_index}.npy")
        batch_inputs = [batch_rows[input_field] for input_field in self.input_fields]
        batch_targets = batch_rows[self.target_field]
        return batch_inputs, batch_targets
    

class FeederProf(DataFeeder):
    """Curriculum creator.

    Takes a pre-trained model and estimates the 'difficulty' of each record.
    Then generates the batches with increasing difficulty.
    """

    def __init__(
        self, trained_model, data_folder, difficulty_levels=5, n_of_epochs=20, **datafeeder_kwargs
    ):

        # Initializes itself as a vanilla DataFeeder
        # with shuffling turned off since scoring doesn't need it
        print(
            f"Initializing [green]prof[/green] with model [green]{trained_model}[/green] and data [red]{data_folder}[/red] "
        )
        datafeeder_kwargs["shuffle"] = False

        self.epoch = 0
        self.n_of_epochs = n_of_epochs

        super().__init__(data_folder, **datafeeder_kwargs)

        self.model_folder = trained_model
        self.model = load_model(trained_model)
        self.savefile = (
            f"{self.model_folder}/{PROF_SAVEFILE}"  # Saves (true-predicted) data
        )

        # Creates an empty array for the scores
        # That is long as the dataset
        self.difficulty_levels = difficulty_levels
        self.is_data_scored = False  # Flag to score data only once
        self._teaching_level = 0  # Minimum level of lessons given

        # Gets the data score
        self.score_data()

        # Overrides __getitem__ method in runtime since the student
        # __getitem__ is no longer required
        # (special methods are called by class, not by instance)
        ## NOTE: This overrides the class method from the first initialization onwards
        # So stick to the if- else version
        # FeederProf.__getitem__ = FeederProf.__getitem_override__

    def _getitem_override(self, batch_index):
        """Gives one batch of data but sorted in ascending order of difficulty"""
        # Following the reference article, increase the size of the data from which
        # the batch is sampled, increasing difficulty

        # Generate data
        net_input, net_target = self.data_generation(self.epoch_records[batch_index])

        # Save the indexes of the batch
        self.last_batch_indexes = np.array(indexes)

        return net_input, net_target

    def on_epoch_end(self, *args):
        """Since on_epoch_end is called without args, use a counter to get epoch number"""
        self.epoch += 1
        # Before next epoch begins the order of the file that will be
        # feeded in the net is chosen
        self.epoch_records = self.datum_indexes.copy()
        self.epoch_records = self.epoch_records[: int(self.epoch/self.n_of_epochs * self.data_len)]
        np.random.shuffle(self.epoch_records)
        print(f"Next files that will be feeded are {self.epoch_records}")
 
    def september(self):
        self.epoch = 0

    def __getitem__(self, batch_index):
        if self.is_data_scored:
            return self._getitem_override(batch_index)
        else:
            return super().__getitem__(batch_index)

    @property
    def teaching_level(self):
        return self._teaching_level

    @teaching_level.setter
    def teaching_level(self, value):
        if not isinstance(value, int):
            raise ValueError("Difficulty must be an integer")
        if value >= self.difficulty_levels or value < 0:
            raise ValueError(f"Difficulty must be 0 < lvl < {self.difficulty_levels}")
        print(
            f"Teaching level set to {value}"
            f" ({len(self.scores[self.scores >= value])/len(self.scores)*100 :.0f}% of samples available)"
        )
        self._teaching_level = value
        self.data_len = len(self.scores[self.scores >= value])

    def score_data(self):
        """Estimates the difficulty of the data.

        Associates the indexes of the batch to a given difficulty score.
        This function must be called BEFORE the __getitem__ override, as it uses ``self`` as
        the generator.
        """
        print("Scoring data..")

        if self.is_data_scored:
            raise RuntimeError(
                "Prof scores are already generated and __getitem__ method is overriden"
            )

        # Tries to load errors
        if not self.load_knowledge():

            # Gets the prof model estimates for the batch
            print("[red]getting true values..[/red]")
            self._true_vals = np.array([batch[1] for batch in track(self)]).reshape(
                (-1)
            )
            print("[red]getting estimates..[/red]")
            self._estimates = self.model.predict(
                self, verbose=1, batch_size=self.batch_size
            ).squeeze()
            self.save_knowledge()

        # Estimates the difficulty of the batch entries
        # From how much the prof model fails on the predictions
        self.errors = np.abs(self._estimates - self._true_vals)

        # Sort everything in ascending order of erroneous prediction
        self.sort_order = np.argsort(self.errors)
        self.datum_indexes = self.datum_indexes[self.sort_order]
        self.errors = self.errors[self.sort_order]
        self._estimates = self._estimates[self.sort_order]
        self._true_vals = self._true_vals[self.sort_order]

        # Since some data is cutted of from prediction because of batch rounding
        # the scored data is less than the original set
        self.data_len = len(self.errors)
        self.scores = np.zeros(self.data_len)

        # Now splits in ``self.difficulty_levels`` equal parts
        # of increasing difficulty
        lvl_dim = len(self.errors) // self.difficulty_levels
        for lvl in range(self.difficulty_levels):
            self.scores[lvl * lvl_dim : (lvl + 1) * lvl_dim] = lvl

        self.is_data_scored = True
        print(f"Prof [green]{self.model_folder}[/green] initialized")

    def save_knowledge(self):
        print(f"[green]Saving[/green] knowledge ({self.savefile})")
        np.save(self.savefile, np.stack((self._true_vals, self._estimates)))

    def load_knowledge(self):
        if exists(self.savefile):
            self._true_vals, self._estimates = np.load(self.savefile)
            print("Errors [blue]loaded[/blue] from file")
            return True
        else:
            return False

    def __len__(self):
        # Cutting the dataset by difficulty shortens it
        return int(np.floor(self.data_len / self.batch_size))
