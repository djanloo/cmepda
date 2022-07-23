"""A context for importing modules.

Inside pre-post-proc use

    >>> from .context import ...

"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../cloudatlas"))
)

from cloudatlas.datafeeders import DataFeeder, FeederProf
from cloudatlas.nets import LstmEncoder, ToaEncoder, TimeSeriesLSTM
import cloudatlas.constants as constants
import cloudatlas.utils as utils
import ppproc.stats as stats
from ppproc.augmentation import Augment
