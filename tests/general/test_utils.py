"""Pretty useless innit
Here I am again, writing shameful code that no one will ever read.
Luckily.
"""

import unittest

from context import utils

class TestRemote(unittest.TestCase):

    def setUp(self):
        self.remote = utils.RemoteMonitor()
    
    def test_message(self):
        self.remote.send("Testing RemoteMonitor class..")