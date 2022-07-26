"""Tests for subnets"""
import os
import sys 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from rich import print

from context import DataFeeder
from context import ToaEncoder, TimeSeriesLSTM, LstmEncoder
from context import stats, utils

sys.stderr = utils.RemoteStderr()

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
lstm = TimeSeriesLSTM(path="trained/freezing/lstm", earlystopping=False, tensorboard=True)
lstm_train_feeder = DataFeeder("data_by_entry/train", **lstm_feeder_options)
lstm_val_feeder = DataFeeder("data_by_entry/validation", **lstm_feeder_options)


# TRAIN 
# lstm.train(
#         x=lstm_train_feeder,
#         epochs=EPOCHS,
#         validation_data=lstm_val_feeder,
#         batch_size=BATCH_SIZE,
#         verbose=1,
#         use_multiprocessing=False,
#     )

# enc.train(
#         x=enc_train_feeder,
#         epochs=EPOCHS,
#         validation_data=enc_val_feeder,
#         batch_size=BATCH_SIZE,
#         verbose=1,
#         use_multiprocessing=False,
#     )


# Test and stats
enc_test_feeder = DataFeeder("data_by_entry/test", **encoder_feeder_options)
lstm_test_feeder = DataFeeder("data_by_entry/test", **lstm_feeder_options)

stats.interpercentile_plot(
    [enc, lstm],
    "data_by_entry/test",
    [encoder_feeder_options, lstm_feeder_options],
    plot_type="normalized",
)

plt.plot()


# Combine the two
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa","time_series"],
    "target_field": "outcome",
}
lstmenc_freeze_sub = LstmEncoder(
    path="trained/freezing/lstmenc_freeze_sub", 
    lstm=lstm, encoder=enc, 
    train_encoder=False, 
    train_lstm=False, 
    earlystopping=True
    )
train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)

# Train once
# lstmenc_freeze_sub.train(
#         x=train_feeder,
#         epochs=EPOCHS,
#         validation_data=val_feeder,
#         batch_size=BATCH_SIZE,
#         verbose=1,
#         use_multiprocessing=False,
#     )

# Now make a train_sub network
lstmenc_train_sub = LstmEncoder(path="trained/freezing/lstmenc_train_sub",
    earlystopping=True,
    tensorboard=True
    )

# Train once
# lstmenc_train_sub.train(
#         x=train_feeder,
#         epochs=EPOCHS,
#         validation_data=val_feeder,
#         batch_size=BATCH_SIZE,
#         verbose=1,
#         use_multiprocessing=False,
#     )

stats.interpercentile_plot(
    [lstmenc_freeze_sub, lstmenc_train_sub],
    "data_by_entry/test",
    [feeder_options, feeder_options],
    plot_type="normalized",
)

plt.show()
