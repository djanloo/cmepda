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

# optionsenc
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)

enc= ToaEncoder(path="trained/ToaEncoder")

for layer in enc.model.layers:
    print(f"Enc: {layer.name}")

lstm = TimeSeriesLSTM(path="trained/TimeSeriesLSTM_backup")

for layer in lstm.model.layers:
    print(f"Lstm: {layer.name}")

lstmenc = LstmEncoder(earlystopping=True, tensorboard=True,
                        encoder=enc, lstm=lstm)

lstmenc.train(
    x=train_feeder,
    epochs=EPOCHS,
    validation_data=val_feeder,
    batch_size=BATCH_SIZE,
    verbose=1,
    use_multiprocessing=False,
)

# w = enc.model.layers[-2].get_weights()[0]
# b = enc.model.layers[-2].get_weights()[1]
# print(w)
# print(b)

