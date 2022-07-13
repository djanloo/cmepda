import os
from tensorflow import keras
from keras.utils.vis_utils import plot_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from context import FeederProf, DataFeeder
from context import LstmEncoder, ToaEncoder, TimeSeriesLSTM

from matplotlib import pyplot as plt

# constants
EPOCHS = 50
BATCH_SIZE = 128

# options
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": "time_series",
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)

# initializing LstmEncoder class
lstmenc = LstmEncoder()
plot_model(lstmenc.model, to_file="assets/lstmencoder.png", show_shapes=True, show_layer_activations=True, show_layer_names=False)

# initializing TimeSeries class
lstm = TimeSeriesLSTM()
plot_model(lstm.model, to_file="assets/lstm.png",  show_shapes=True, show_layer_activations=True,show_layer_names=False)

enc = ToaEncoder()
plot_model(enc.model, to_file="assets/encoder.png", show_shapes=True, show_layer_activations=True,show_layer_names=False)


# TensorBoard callbacks, # Write TensorBoard logs to `./logs` directory
tb_callbacks = keras.callbacks.TensorBoard(
    log_dir=f"{enc.path}/logs", histogram_freq=1
)

enc.train(
    x=train_feeder,
    epochs=EPOCHS,
    validation_data=val_feeder,
    batch_size=BATCH_SIZE,
    callbacks=[tb_callbacks],
    verbose=1,
    use_multiprocessing=False,
)
