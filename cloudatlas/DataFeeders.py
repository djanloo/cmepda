"""Module for data feeders"""
from calendar import firstweekday
from http.client import NOT_IMPLEMENTED
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

# Turn off keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DataFeederKeras(keras.utils.Sequence):
    def __init__(
        self, folder, batch_size=32, shuffle=True, input_fields=None, target_field=None
    ):

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
        return int(np.floor( self.data_len / self.batch_size))

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

class FeederProf(DataFeederKeras):
    """Curriculum creator"""

    def __init__(self, trained_model, data_folder, difficulty_levels = 5, 
                    **datafeeder_kwargs):

        # Initializes itself as a vanilla DataFeeder
        # with shuffling turned off since scoring doesn't need it
        print(f"Initializing [green]prof[/green] with model [green]{trained_model}[/green] and data [red]{data_folder}[/red] ")
        datafeeder_kwargs['shuffle'] = False
        super().__init__(data_folder,**datafeeder_kwargs )

        self.model_folder = trained_model
        self.model = load_model(trained_model)

        # Creates an empty array for the scores
        # That is long as the dataset
        self.difficulty_levels = difficulty_levels
        self.scores = np.empty(self.data_len)
        self.scores[:] = np.nan
        self.is_data_scored = False # Flag to score data only once
        self._learning_level = 0 # Minimum level of lessons given
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

        # Cuts the dataset by difficulty
        restricted_file_indexes = self.datum_indexes[self.scores >= self.learning_level]

        # Selects the indexes inside the restricted dataset
        indexes_of_file_indexes = np.random.randint(0, (batch_index + 1) * self.batch_size, size=self.batch_size)
        
        # Get the file indexes of the selected data
        indexes = restricted_file_indexes[indexes_of_file_indexes]

        # Generate data
        net_input, net_target = self.data_generation(indexes)

        # Save the indexes of the batch
        self.last_batch_indexes = np.array(indexes)

        return net_input, net_target

    def __getitem__(self, batch_index):
        if self.is_data_scored:
            return self._getitem_override(batch_index)
        else:
            return super().__getitem__(batch_index)

    @property
    def learning_level(self):
        return self._learning_level
    
    @learning_level.setter
    def learning_level(self, value):
        if not isinstance(value, int):
            raise ValueError("Difficulty must be an integer")
        if value >= self.difficulty_levels or value < 0:
            raise ValueError(f"Difficulty must be 0 < lvl < {self.difficulty_levels}")
        print(f"Learning level set to {value}" \
              f" ({len(self.scores[self.scores >= value])/len(self.scores)*100 :.0f}% of samples available)" )
        self._learning_level = value
        self.data_len = len(self.scores[self.scores >= value])

    def pacing(self, epoch):
        raise NotImplementedError("prof pacing function is user defined")
    
    def scoring(self, errors):
        #raise NotImplementedError("prof scoring function is user defined")
        mean_error = np.mean(np.abs(errors))
        return  np.log(np.abs(errors/mean_error) + 1.0)
    
    def _normalize_scores(self):
        """Generates the difficulty label from the score value lying in [0, 1]"""
        self.scores = self.scores - np.min(self.scores)
        self.scores /= np.max(self.scores)

        # Multiply for the number of levels so that (n lvls = 5)
        # e.g. score = 0.3 -> score = floor(5*0.3) = 1
        # e.g. score = 0.9 -> score = floor(5*0.9) = 4 
        self.scores = np.floor(self.difficulty_levels*self.scores)
        plt.show()

    def score_data(self):
        """Estimates the difficulty of the data.

        Associates the indexes of the batch to a given difficulty score.
        This function must be called BEFORE the __getitem__ override, as it uses ``self`` as
        the generator.
        """
        print("Scoring data..")

        if self.is_data_scored:
            raise RuntimeError("Prof scores are already generated and __getitem__ method is overriden")
        
        # Tries to load errors
        if not self.load_errors():

            # Gets the prof model estimates for the batch
            print("[red]getting true values..[/red]")
            true_vals = np.array([batch[1] for batch in track(self)]).reshape((-1))
            print("[red]getting estimates..[/red]")
            estimates = self.model.predict(self, verbose=1, batch_size=self.batch_size).squeeze()

            # Estimates the difficulty of the batch entries
            # From how much the prof model fails on the predictions
            self.errors = estimates - true_vals
            self.save_errors()

        # Set the scores of the data using the dataset indexes
        self.scores[self.datum_indexes[:len(self.errors)]] = self.scoring(self.errors)
        
        # Removes unscored data (that is still nan)
        self.datum_indexes = self.datum_indexes[np.logical_not(np.isnan(self.scores))]
        self.scores = self.scores[np.logical_not(np.isnan(self.scores))]

        self._normalize_scores()
        
        # Sort everything in ascending order of difficulty
        sort_order = np.argsort(self.scores)
        self.datum_indexes = self.datum_indexes[sort_order]
        self.errors = self.errors[sort_order]
        self.scores= self.scores[sort_order]

        self.is_data_scored = True
        print(f"Prof [green]{self.model_folder}[/green] initialized")

    def save_errors(self):
        print(f"[green]Saving[/green] errors ({self.model_folder}/prof_errors.npy)")
        np.save(f"{self.model_folder}/prof_errors.npy", self.errors)

    def load_errors(self):
        if exists(f"{self.model_folder}/prof_errors.npy"):
            self.errors = np.load(f"{self.model_folder}/prof_errors.npy")
            if np.isnan(self.errors).any():
                print(f"Errors are nan. Deleting file..")
                os.remove(f"{self.model_folder}/prof_errors.npy")
                return False
            print("Errors [blue]loaded[/blue] from file")
            return True
        else:
            return False
    
    def __len__(self):
        # Cutting the dataset by difficulty shortens it
        return int(np.floor( self.data_len / self.batch_size))


    