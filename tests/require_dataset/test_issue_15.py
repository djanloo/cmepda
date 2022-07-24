"""Tests for subnets"""
import os
import multiprocessing as mp

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
enc = ToaEncoder(path="trained/freezing/enc", earlystopping=True, tensorboard=True)
enc_train_feeder = DataFeeder("data_by_entry/train", **encoder_feeder_options)
enc_val_feeder = DataFeeder("data_by_entry/validation", **encoder_feeder_options)

# LSTM subnet
lstm_feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": "time_series",
    "target_field": "outcome",
}
lstm = TimeSeriesLSTM(path="trained/freezing/lst", earlystopping=True, tensorboard=True)
lstm_train_feeder = DataFeeder("data_by_entry/train", **lstm_feeder_options)
lstm_val_feeder = DataFeeder("data_by_entry/validation", **lstm_feeder_options)


# TRAIN
# Use multiprocessing to train both at the same time
# Used to multiprocess training
def train_subnet(net, train_feeder, val_feeder):
    net.train(
        x=train_feeder,
        epochs=EPOCHS,
        validation_data=val_feeder,
        batch_size=BATCH_SIZE,
        verbose=1,
        use_multiprocessing=False,
    )

pool = mp.Pool(processes=4)
args = (
    (enc, enc_train_feeder, enc_val_feeder),
    (lstm, lstm_train_feeder, lstm_val_feeder),
)
pool.map(train_subnet, args)

# Test and stats
enc_test_feeder = DataFeeder("data_by_entry/test", **encoder_feeder_options)
lstm_test_feeder = DataFeeder("data_by_entry/test", **lstm_feeder_options)

stats.interpercentile_plot(
    [enc, lstm],
    "data_by_entry/test",
    [encoder_feeder_options, lstm_feeder_options],
    relative_error=False,
)

plt.show()
