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
lstm = TimeSeriesLSTM(path="trained/freezing/lst_2")
lstmenc = LstmEncoder(path="lstmenc_train_sub")

lstmenc_notrain = LstmEncoder(
    path="trained/lstmenc",
    earlystopping=True,
    tensorboard=True,
    encoder=enc,
    lstm=lstm,
)

fig, axes = plt.subplots(2, 2)

w_notrain = lstmenc_notrain.model.get_layer(name="lstmenc_dense_1").get_weights()[0]
b_notrain = lstmenc_notrain.model.get_layer(name="lstmenc_dense_1").get_weights()[1]

w_train = lstmenc.model.get_layer(name="lstmenc_dense_1").get_weights()[0]
b_train = lstmenc.model.get_layer(name="lstmenc_dense_1").get_weights()[1]

breakpoint()

w1 = axes[0, 0].imshow(w_notrain)
plt.colorbar(w1, ax=axes[0, 0])
axes[0, 0].set_title("Freeze-sub weights ")
axes[0, 0].axis("off")

b1 = axes[0, 1].imshow([b_notrain])
plt.colorbar(b1, ax=axes[0, 1])
axes[0, 1].set_title("Freeze-sub biases")
axes[0, 1].axis("off")

w2 = axes[1, 0].imshow(w_train)
plt.colorbar(w2, ax=axes[1, 0])
axes[1, 0].set_title("Train-sub weights ")
axes[1, 0].axis("off")

b2 = axes[1, 1].imshow([b_train])
plt.colorbar(b2, ax=axes[1, 1])
axes[1, 1].set_title("Train-sub biases")
axes[1, 1].axis("off")


plt.show()
# print(w)
# print(b)
