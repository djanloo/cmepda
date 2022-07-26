"""A context for importing modules.

    >>> from context import ...

"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../cloudatlas"))
)

from cloudatlas.datafeeders import DataFeeder, FeederProf
from cloudatlas.nets import LstmEncoder, ToaEncoder, TimeSeriesLSTM, LinearProbe
import cloudatlas.constants as constants
import cloudatlas.utils as utils
import ppproc.stats as stats
