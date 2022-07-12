import os
import datetime
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datafeeders import FeederProf, DataFeeder
from net import LstmEncoder

from matplotlib import pyplot as plt

# constants
EPOCHS = 5
BATCH_SIZE = 128

# options
feeder_options = {
    "shuffle": False,  # Only for testing
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry_aug/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)

# initializing LstmEncoder class
claretta = LstmEncoder(path="trained/claretta")

print(claretta.resolution_on(test_feeder))
# exit()

# TensorBoard callbacks, # Write TensorBoard logs to `./logs` directory
tb_callbacks = keras.callbacks.TensorBoard(log_dir='trained/claretta/logs',
                                           histogram_freq=1)

claretta.train(
    x=train_feeder,
    epochs=EPOCHS,
    validation_data=val_feeder,
    batch_size=BATCH_SIZE,
    callbacks=[tb_callbacks],
    verbose=1,
    use_multiprocessing=False,
)
