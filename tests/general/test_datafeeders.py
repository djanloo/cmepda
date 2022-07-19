"""Creates a dummy dataset then checks datafeeders"""
import os
from shutil import rmtree
import unittest
import numpy as np

from context import DataFeeder

DUMMY_SHAPE = (3,4)
N = 100

DATASET_DTYPE = [("input", np.float32, DUMMY_SHAPE),
                 ("output", np.float32)]

class TestSISO(unittest.TestCase):
    """Single input single output test"""
    def setUp(self):
        dataset = np.empty(N, dtype=DATASET_DTYPE)
        dataset["input"] = np.random.uniform(0, 1, size=(N,) + DUMMY_SHAPE)
        dataset["output"] = np.random.uniform(-1, 0, size=(N,))
        
        # Splits the dataset in files
        if not os.path.exists("dataset_folder"):
            os.mkdir("dataset_folder")
        
        for i, record in enumerate(dataset):
            np.save(f"dataset_folder/part_{i}.npy", record)
        
    def test_zclearup(self):
        rmtree("dataset_folder")
        