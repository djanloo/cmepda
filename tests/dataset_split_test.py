import sys, os
from os.path import join
from shutil import rmtree
import numpy as np
import unittest

# Make possible importing modules from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from cloudatlas.utils import DataFeeder, animate_time_series, split_dataset

TOA_SHAPE = (9,9,1)
TS_SHAPE = (9,9,80)
N = 100 # Fake dataset number of events, keep it low
PARENT_FOLDER = "test_splitted_data"

class dataSplitTest(unittest.TestCase):
    """Class to test data split.
    
    It also generates the fake dataset to test datafeeding.
    """
    def setUp(self):
        self.time_of_arrival = np.random.uniform(0, 1, size=N*9*9).reshape((-1,) + TOA_SHAPE)
        self.time_series = np.random.uniform(0, 1, size=N*9*9*80).reshape((-1,) + TS_SHAPE)
        self.outcomes = np.random.uniform(0, 1, size=N)
    
    def test_split(self):
        split_dataset((self.time_of_arrival, self.time_series, self.outcomes),
                        "fake_dataset",
                        ["toa", "time_series", "outcomes"],
                        10,
                        parent=PARENT_FOLDER # Required form permission issues
                        )

class datafeederTest(unittest.TestCase):

    def setUp(self):
        self.test_data = DataFeeder(join(PARENT_FOLDER, "fake_dataset_splitted_data")).feed()
        self.block = next(self.test_data)

    def test_order(self):
        # Check for bug #2
        # Since the order is toa, ts, outcome shapes must be
        self.assertEqual(self.block["toa"].shape[1:], (9, 9, 1))
        self.assertEqual(self.block["time_series"].shape[1:], (9, 9, 80))

    def test_anim(self):
        # Animation is not shown, only checks for errors
        u = animate_time_series(self.block["time_series"][0])
    
    def test_zcleanup(self):
        # Deletes the fake dataset from memory
        # The 'z' in the method name is a workaround to alphabetical
        # test oredering
        rmtree(PARENT_FOLDER)
