import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from matplotlib import rcParams
from rich import print
import numpy as np

from context import FeederProf, DataFeeder
from context import LstmEncoder, ToaEncoder, TimeSeriesLSTM


# constants
EPOCHS = 100
BATCH_SIZE = 128

rcParams["font.family"] = "serif"
rcParams["font.size"] = 10

# optionsenc
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

train_feeder = DataFeeder("/Users/luciapapalini/Desktop/cmepda/data_by_entry_aug/train", **feeder_options)
val_feeder = DataFeeder("/Users/luciapapalini/Desktop/cmepda/data_by_entry_aug/validation", **feeder_options)
test_feeder = DataFeeder("/Users/luciapapalini/Desktop/cmepda/data_by_entry_aug/test", **feeder_options)

lstm = TimeSeriesLSTM()

