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
        # This should not be a problem

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
        net_input, net_target = self.__data_generation(indexes)

        # Test for curriculum learning: save the indexes of the batch
        self.last_batch_indexes = np.array(indexes)

        return net_input, net_target

    def on_epoch_end(self):
        """Shuffles indexes after each epoch"""
        self.datum_indexes = np.arange(self.data_len)
        if self.shuffle:
            print(f"[blue]Shuffled indexes[/blue] in DataFeederKeras({self.folder})")
            np.random.shuffle(self.datum_indexes)

    def __data_generation(self, batch_datum_indexes):
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

    def __init__(self, trained_model, data_folder, 
                    uniform_pacing=True, difficulty_levels = 5, 
                    **datafeeder_kwargs):

        # Initializes itself as a vanilla DataFeeder
        # with shuffling turned off since that scoring doesn't need shuffling
        print(f"Initializing [green]prof[/green] with model [green]{trained_model}[/green] and data [red]{data_folder}[/red] ")
        datafeeder_kwargs['shuffle'] = False
        super().__init__(data_folder,**datafeeder_kwargs )
        self.model_folder = trained_model
        self.model = load_model(trained_model)

        if uniform_pacing:
            # Curriculum with no pacing: each batch has the same size
            self.pacing = lambda i: self.batch_size # this is set in super().__init__()
        
        # Uses itself as a feeder to model predict
        # At this point the generation procedure is the one in
        # vanilla DataFeederKeras, given by super()
        # print("Getting average prof error.. ")        # Unnecessary ?
        # self.avg_error = self.model.evaluate(self)[1] # Unnecessary ?

        # Creates an empty array for the scores
        # That is long as the dataset
        self.difficulty_levels = difficulty_levels
        self.scores = np.empty(self.data_len)
        self.is_data_scored = False # Flag to score data only once7

        # Tries to load scores
        if not self.load_scores():
            self.score_data()

        # Overrides __getitem__ method in runtime since the student 
        # __getitem__ is no longer required
        # (special methods are called by class, not by instance)
        FeederProf.__getitem__ = FeederProf.__getitem_override__

    def __getitem_override__(self, batch_index):
        raise RuntimeError("you genius")
        
    def pacing(self, epoch):
        raise NotImplementedError("prof pacing function is user defined")
    
    def scoring(self, errors):
        #raise NotImplementedError("prof scoring function is user defined")
        return errors
    
    def _normalize_scores(self):
        """Normalizes the scores in the [0,1] interval then generates levels"""
        self.scores = self.scores - np.min(self.scores)
        self.scores /= np.max(self.scores)

        # Multiply for the number of levels so that
        # e.g. score = 0.3 -> score = floor(5*0.3) = 1
        # e.g. score = 0.9 -> score = floor(5*0.9) = 4 
        self.scores = np.floor(self.difficulty_levels*self.scores)

    def score_data(self):
        print("Scoring data..")
        if self.is_data_scored:
            raise RuntimeError("Prof scores are already generated and __getitem__ method is overriden")
        """Estimates the difficulty of the data.

        Associates the indexes of the batch to a given difficulty score
        """

        # Gets the prof model estimates for the batch
        print("[red]getting true values..[/red]")
        true_vals = np.array([batch[1] for batch in track(self)]).reshape((-1))
        print("[red]getting estimates..[/red]")
        estimates = self.model.predict(self, verbose=1, batch_size=self.batch_size).squeeze()

        # Estimates the difficulty of the batch entries
        # From how much the prof model fails on the predictions
        difficulties = self.scoring(estimates - true_vals)

        # Set the scores of the data using the dataset indexes
        self.scores[self.datum_indexes[:len(estimates)]] = difficulties
        self._normalize_scores()
        self.is_data_scored = True
        self.save_scores()
        print(f"Prof [green]{self.model_folder}[/green] initialized")

    def save_scores(self):
        np.save(f"{self.model_folder}/prof_scores.npy", self.scores)

    def load_scores(self):
        if exists(f"{self.model_folder}/prof_scores.npy"):
            self.scores = np.load(f"{self.model_folder}/prof_scores.npy")
            print("Scores [blue]loaded[/blue] from file")
            self.is_data_scored = True
            return True
        else:
            return False
