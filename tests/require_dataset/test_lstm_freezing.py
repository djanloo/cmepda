"""Aim of this script is to estimate the improvement due to the presence of the encoder
in the LstemEncoder network.

To do so, it is necessary to pre-train an lstm subnet and check its resolution.

Then a complete net is built using the previously trained lstm subnet.
Finally the complete net undergoes a train stage and resolution is estimated.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from matplotlib import pyplot as plt
from matplotlib import rcParams
from rich import print

from context import DataFeeder
from context import LstmEncoder , TimeSeriesLSTM, utils
from context import stats

# Sends a message if something has gone wrong
sys.stderr = utils.RemoteStderr()

# constants
EPOCHS = 50
BATCH_SIZE = 128

# Directories
# work_dir = os.getcwd()
# parent_dir = os.path.dirname(work_dir)  # I go up twice
# parent_dir = os.path.dirname(parent_dir)
# os.chdir(parent_dir)


rcParams["font.family"] = "serif"
rcParams["font.size"] = 10

#%% Part 1: subnet train

# Prepare datafeeder: lstm does not have toa matrices as input
lstm_feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": "time_series",
    "target_field": "outcome",
}

# Then train and validation feeders are created
lstm_train_feeder = DataFeeder("data_by_entry/train", **lstm_feeder_options)
lstm_val_feeder = DataFeeder("data_by_entry/validation", **lstm_feeder_options)

lstm = TimeSeriesLSTM(
    path="trained/test_freezing_lstm", earlystopping=True, tensorboard=True
)
# lstm.train(
#     x=lstm_train_feeder,
#     epochs=EPOCHS,
#     validation_data=lstm_val_feeder,
#     batch_size=BATCH_SIZE,
#     verbose=1,
#     use_multiprocessing=False,
# )

# Resolution estimation
lstm_test_feeder = DataFeeder("data_by_entry/test", **lstm_feeder_options)
# print(f"Only lstm res is {lstm.resolution_on(lstm_test_feeder)}")


#%% Part 2: build and check global network

# Prepare global datafeeder: now i need both toa and ts
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

# Create the lsmEncoder datafeeders
train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)

# Build the net
lstmenc = LstmEncoder(
    path="trained/test_lstm_freezing_lstmencoder",
    lstm=lstm, train_lstm=False, # Uses the lstm of line 42  # But then freezes it
    earlystopping=True, tensorboard=True,
)

# Now train the net (lstm subnet excluded)
# lstmenc.train(
#     x=train_feeder,
#     epochs=EPOCHS,
#     validation_data=val_feeder,
#     batch_size=BATCH_SIZE,
#     verbose=1,
#     use_multiprocessing=True,
# )

# Then checks resolution
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)
# print(f"LSTM + encoder res is {lstmenc.resolution_on(test_feeder)}")

stats.interpercentile_plot(
    [lstm, lstmenc],
    "data_by_entry/test",
    [lstm_feeder_options, feeder_options],
    relative_error=True,
)

plt.show()