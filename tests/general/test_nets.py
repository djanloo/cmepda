"""Tests if the networks are successfully build"""
import unittest
from context import ToaEncoder, TimeSeriesLSTM, LstmEncoder

class TestBuild(unittest.TestCase):

    def test_build_toa(self):
        self.net = ToaEncoder()
    
    def test_build_lstm(self):
        self.net = TimeSeriesLSTM()
    
    def test_build_lstmenc(self):
        self.net = LstmEncoder()