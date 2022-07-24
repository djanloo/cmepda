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
from context import ToaEncoder, utils
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
encoder_feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": "toa",
    "target_field": "outcome",
}

# Then train and validation feeders are created
encoder_train_feeder = DataFeeder("data_by_entry/train", **encoder_feeder_options)
encoder_val_feeder = DataFeeder("data_by_entry/validation", **encoder_feeder_options)

enc = ToaEncoder(
    path="trained/test_encoder", earlystopping=False, tensorboard=True
)

enc.train(
     x=encoder_train_feeder,
     epochs=EPOCHS,
     validation_data=encoder_val_feeder,
     batch_size=BATCH_SIZE,
     verbose=1,
     use_multiprocessing=False,
 )

# Resolution estimation
encoder_test_feeder = DataFeeder("data_by_entry/test", **encoder_feeder_options)
# print(f"Only lstm res is {lstm.resolution_on(lstm_test_feeder)}")
