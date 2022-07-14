import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from rich import print

from context import FeederProf, DataFeeder
from context import LstmEncoder, ToaEncoder, TimeSeriesLSTM

# constants
EPOCHS = 50
BATCH_SIZE = 128

# Encoder subnet

# options for feeders
# Note that the input field is different for the subnets
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": "toa",
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)

# initializing TimeSeries class
enc = ToaEncoder(earlystopping=True, tensorboard=True)

enc.train(
    x=train_feeder,
    epochs=EPOCHS,
    validation_data=val_feeder,
    batch_size=BATCH_SIZE,
    verbose=1,
    use_multiprocessing=False,
)

# LSTM subnet
feeder_options["input_fields"] = "time_series"

train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)

lstm = TimeSeriesLSTM(earlystopping=True, tensorboard=True)

lstm.train(
    x=train_feeder,
    epochs=EPOCHS,
    validation_data=val_feeder,
    batch_size=BATCH_SIZE,
    verbose=1,
    use_multiprocessing=False,
)
