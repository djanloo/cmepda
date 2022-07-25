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

# enc = ToaEncoder(path="trained/freezing/enc")
# lstm = TimeSeriesLSTM(path="trained/freezing/lstm")
lstmenc_notrain = LstmEncoder(
    path="trained/freezing/lstmenc_freeze_sub",
    earlystopping=True,
    tensorboard=True,
)
lstmenc = LstmEncoder(path="trained/freezing/lstmenc_train_sub")


w_notrain = lstmenc_notrain.model.get_layer(name="lstmenc_dense_1").get_weights()[0]
b_notrain = lstmenc_notrain.model.get_layer(name="lstmenc_dense_1").get_weights()[1]

w_train = lstmenc.model.get_layer(name="lstmenc_dense_1").get_weights()[0]
b_train = lstmenc.model.get_layer(name="lstmenc_dense_1").get_weights()[1]

fig, ax = plt.subplots(2,1)

vmax = np.max(np.stack((w_train, w_notrain)))
vmin = np.min(np.stack((w_train, w_notrain)))

notrain_image = ax[0].imshow(w_notrain.T, vmin=vmin, vmax=vmax)
train_image = ax[1].imshow(w_train.T, vmin=vmin, vmax=vmax )

ax[0].axis("off")
ax[1].axis("off")

plt.show()
# print(w)
# print(b)
