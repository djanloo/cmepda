"""Creates a dummy dataset then checks augmentation"""
import os
from shutil import rmtree
import unittest
import numpy as np

from context import Augment
from context import constants
from context import FeederProf

class TestAUG(unittest.TestCase):
    """Single input augmentation"""

    def setUp(self):
        # matrix to test
        self.matrix = np.arange(81).reshape(9,9)
        
        # keys of various types of augmentation
        aug_types = ["rot", "flip_lr", "flip_ud", "flip_diag"]
        
        # initialize prof to call Augment
        feeder_options = {
            "batch_size": 100,
            "input_fields": ["toa", "time_series"],
            "target_field": "outcome",
        }

        # Calling prof Albertino
        prof_train = FeederProf(
            "trained/albertino",
            constants.DIR_DATA_BY_ENTRY_AUG + "/test",
            **feeder_options,
            n_of_epochs=1,
        )

        # initialize and run augmentation
        aug = Augment(prof_train)
        self.augmented_mat = aug.augment_matrix(self.matrix)
        
    def test_rot(self):
        self.assertEqual(np.rot90(self.matrix), self.augmented_mat['rot'])
        
    def test_flip_lr(self):
        self.assertEqual(np.fliplr(self.matrix), self.augmented_mat['flip_lr'])
        
    def test_flip_ud(self):
        self.assertEqual(np.flipud(self.matrix), self.augmented_mat['flip_ud'])
        
    def test_flip_diag(self):
        self.assertEqual(np.transpose(self.matrix), self.augmented_mat['flip_diag'])
