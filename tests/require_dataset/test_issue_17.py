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

train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)

enc = ToaEncoder(path="trained/freezing/enc")
lstm = TimeSeriesLSTM(path="trained/freezing/lst")
lstmenc_notrain = LstmEncoder(
    path="trained/lstmenc",
    earlystopping=True,
    tensorboard=True,
)
lstmenc = LstmEncoder(path="lstmenc_train_sub")


w_notrain = lstmenc_notrain.model.get_layer(name="lstmenc_dense_1").get_weights()[0]
b_notrain = lstmenc_notrain.model.get_layer(name="lstmenc_dense_1").get_weights()[1]

w_train = lstmenc.model.get_layer(name="lstmenc_dense_1").get_weights()[0]
b_train = lstmenc.model.get_layer(name="lstmenc_dense_1").get_weights()[1]

w_notrain = np.swapaxes(w_notrain, 0, 1)
mean = np.mean(w_notrain, axis=0)
plt.plot(mean)
plt.title("Mean of the (16, 10_000) weight matrix")

plt.figure(2)
for row in w_notrain:
    plt.plot(row, alpha=0.2)
plt.title("Weights cell_by_cell")
plt.show()
# print(w)
# print(b)
