import os
import datetime
from pyexpat import model
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datafeeders import FeederProf, DataFeeder
from net import LstmEncoder, ToaEncoder, TimeSeriesLSTM

from matplotlib import pyplot as plt

# constants
EPOCHS = 50
BATCH_SIZE = 128

# options
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": "toa",
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)

# initializing LstmEncoder class
model = ToaEncoder(path="trained/toa_encoder")

# TensorBoard callbacks, # Write TensorBoard logs to `./logs` directory
tb_callbacks = keras.callbacks.TensorBoard(
    log_dir=f"{model.path}/logs", histogram_freq=1
)

model.train(
    x=train_feeder,
    epochs=EPOCHS,
    validation_data=val_feeder,
    batch_size=BATCH_SIZE,
    callbacks=[tb_callbacks],
    verbose=1,
    use_multiprocessing=False,
)
