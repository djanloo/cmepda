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
EPOCHS = 50
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

# call the feeders for the different scopes
train_feeder = DataFeeder("data_by_entry_height/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry_height/validation", **feeder_options)
test_feeder = DataFeeder("data_by_entry_height/test", **feeder_options)

# initialize the LstmEncoder
lstmenc = LstmEncoder(
    path="trained/lstmenc_aug",
    earlystopping=False,
    tensorboard=True
)

# Train the net
lstmenc.train(
     x=train_feeder,
     epochs=EPOCHS,
     validation_data=val_feeder,
     batch_size=BATCH_SIZE,
     verbose=1,
     use_multiprocessing=False,
 )

# Resolution estimation
encoder_test_feeder = DataFeeder("data_by_entry_height/test", **feeder_options)
print(f"Whole augmented net is {lstmenc.resolution_on(test_feeder)}")