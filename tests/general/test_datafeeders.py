"""Creates a dummy dataset then checks datafeeders"""
import os
from shutil import rmtree
import unittest
import numpy as np

from context import DataFeeder
from context import constants

DUMMY_SHAPE = (3,4)
N = 100

DATASET_DTYPE_SISO = [("input", np.float32, DUMMY_SHAPE),
                    ("output", np.float32)]

DATASET_DTYPE_MISO = [  ("input1", np.float32, DUMMY_SHAPE),
                        ("input2", np.float32, DUMMY_SHAPE),
                        ("output", np.float32)]

FEEDER_OPTS_SISO =  {
    "shuffle": True,
    "batch_size": 10,
    "input_fields": "input",
    "target_field": "output",
}
FEEDER_OPTS_MISO = {
    "shuffle": True,
    "batch_size": 10,
    "input_fields": ["input1", "input2"],
    "target_field": "output",
}

FOLDER_SISO = "dataset_folder_siso"
FOLDER_MISO = "dataset_folder_miso"

class TestSISO(unittest.TestCase):
    """Single input single output test"""
    def setUp(self):
        dataset = np.empty(N, dtype=DATASET_DTYPE_SISO)
        dataset["input"] = np.random.uniform(0, 1, size=(N,) + DUMMY_SHAPE)
        dataset["output"] = np.random.uniform(-1, 0, size=(N,))
        
        # Splits the dataset in files
        if not os.path.exists(FOLDER_SISO):
            os.mkdir(FOLDER_SISO)
        
        for i, record in enumerate(dataset):
            np.save(f"{FOLDER_SISO}/{constants.FILENAME.format(name=i)}", record)

    def test_feeder(self):
        jimbo = DataFeeder(FOLDER_SISO, **FEEDER_OPTS_SISO)

    def test_zclearup(self):
        rmtree(FOLDER_SISO)



class TestMISO(unittest.TestCase):
    """Single input single output test"""
    def setUp(self):
        dataset = np.empty(N, dtype=DATASET_DTYPE_MISO)
        dataset["input1"] = np.random.uniform(0, 1, size=(N,) + DUMMY_SHAPE)
        dataset["input2"] = np.random.uniform(0, 1, size=(N,) + DUMMY_SHAPE)
        dataset["output"] = np.random.uniform(-1, 0, size=(N,))
        
        # Splits the dataset in files
        if not os.path.exists(FOLDER_MISO):
            os.mkdir(FOLDER_MISO)
        
        for i, record in enumerate(dataset):
            np.save(f"{FOLDER_MISO}/{constants.FILENAME.format(name=i)}", record)

    def test_feeder(self):
        jimbo = DataFeeder(FOLDER_MISO, **FEEDER_OPTS_MISO)

    def test_zclearup(self):
        rmtree(FOLDER_MISO)
        