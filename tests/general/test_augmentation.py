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

        # initialize and run augmentation
        aug = Augment()
        self.augmented_mat = aug.augment_matrix(self.matrix)
        
    def test_rot(self):
        self.assertEqual(np.rot90(self.matrix), self.augmented_mat['rot'])
        
    def test_flip_lr(self):
        self.assertEqual(np.fliplr(self.matrix), self.augmented_mat['flip_lr'])
        
    def test_flip_ud(self):
        self.assertEqual(np.flipud(self.matrix), self.augmented_mat['flip_ud'])
        
    def test_flip_diag(self):
        self.assertEqual(np.transpose(self.matrix), self.augmented_mat['flip_diag'])
