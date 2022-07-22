"""Tests for subnets"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from rich import print

from context import DataFeeder
from context import ToaEncoder, TimeSeriesLSTM
from context import stats

# constants
EPOCHS = 50
BATCH_SIZE = 128

# Encoder subnet

# options for feeders
# Note that the input field is different for the subnets
encoder_feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": "toa",
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry/train", **encoder_feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **encoder_feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **encoder_feeder_options)

# initializing TimeSeries class
enc = ToaEncoder(
    path="trained/encoder_redesigned", earlystopping=True, tensorboard=True
)

# TRAIN ONCE
enc.train(
    x=train_feeder,
    epochs=EPOCHS,
    validation_data=val_feeder,
    batch_size=BATCH_SIZE,
    verbose=1,
    use_multiprocessing=False,
)

# LSTM subnet
lstm_feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": "time_series",
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry/train", **lstm_feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **lstm_feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **lstm_feeder_options)

lstm = TimeSeriesLSTM(earlystopping=True, tensorboard=True)

# TRAIN ONCE
# lstm.train(
#     x=train_feeder,
#     epochs=EPOCHS,
#     validation_data=val_feeder,
#     batch_size=BATCH_SIZE,
#     verbose=1,
#     use_multiprocessing=False,
# )

stats.interpercentile_plot(
    [enc, lstm],
    "data_by_entry/test",
    [encoder_feeder_options, lstm_feeder_options],
    relative_error=False,
)

plt.show()
