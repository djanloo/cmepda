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
        np.testing.assert_allclose(self.augmented_mat['rot'], np.rot90(self.matrix))
        
    def test_flip_lr(self):
        np.testing.assert_allclose(self.augmented_mat['flip_lr'], np.fliplr(self.matrix))
        
    def test_flip_ud(self):
        np.testing.assert_allclose(self.augmented_mat['flip_ud'], np.flip_ud(self.matrix))

    def test_flip_diag(self):
        np.testing.assert_allclose(self.augmented_mat['flip_diag'], np.transpose(self.matrix))