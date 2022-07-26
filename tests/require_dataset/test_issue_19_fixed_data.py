import numpy
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.metrics import RootMeanSquaredError
from rich.progress import track 

from context import LstmEncoder, DataFeeder

try:
    x, y = np.load("w_deleteme.npy")
except:
    feeder_options = {
        "shuffle": False,
        "batch_size": 300,
        "input_fields": ["toa","time_series"],
        "target_field": "outcome",
    }
    lstmenc_freeze_sub = LstmEncoder(
        path="trained/freezing/lstmenc_freeze_sub_test_more_output", 
        train_encoder=False, 
        train_lstm=False, 
        earlystopping=True
        )

    train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
    val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)

    x = lstmenc_freeze_sub.model.predict(train_feeder).squeeze()
    y = np.array([batch[1] for batch in track(train_feeder)]).reshape(-1)

    w = np.stack((x,y))
    np.save("w_deleteme.npy", w)

plt.plot(x, y)

input_layer = Input(shape=(1,))
bb = BatchNormalization()(input_layer)
dense = Dense(16, activation="relu")(input_layer)
dense = Dense(16, activation="relu")(dense)
dense = Dense(16, activation="relu")(dense)
dense = Dense(16, activation="relu")(dense)

dense = Dense(1)(dense)

model = Model(inputs=input_layer, outputs=dense)
model.compile(
            optimizer= "adam",
            loss = "mean_squared_error",
            metrics = [RootMeanSquaredError()],
            )
model.fit(x=x, y=y, epochs=200)
x__ = np.linspace(np.min(x), np.max(x), 100)
plt.plot(x__, model.predict(x__).squeeze(), color="red")
plt.show()