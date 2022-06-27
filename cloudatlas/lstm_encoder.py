"""Test to see whether the lstm+encoder combo is a good idea

For each time series the encoder receives the same time of arrival matrix.
Perhaps this data redundance will overtrain that part of the net.

"""
from genericpath import exists
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import LSTM, Dense, Input, Flatten, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model

import utils
from rich.progress import track
from rich import print

input_toa = Input(shape=(9, 9, 1))
flat = Flatten()(input_toa)
enc = Dense(9, activation="relu")(flat)  
enc = Dense(4, activation="relu")(enc)
enc = Dense(4, activation="relu")(enc)
encoder = Model(inputs=input_toa, outputs=enc)

input_ts = Input(shape=(80,81))
lstm = LSTM(64)(input_ts)
dense = Dense(16, activation='relu')(lstm)
long_short_term_memory = Model(inputs=input_ts, outputs=dense)

conc = concatenate([encoder.output, long_short_term_memory.output])
z = Dense(4, activation="relu")(conc)
z = Dense(4, activation="linear")(z)
z = Dense(1, activation="linear")(z)
global_model = Model(inputs=[encoder.input, long_short_term_memory.input], outputs=z)

global_model.compile(optimizer="adam", loss="mean_squared_error")
plot_model(global_model)

# Load the dataset feeders
test_feeder = utils.DataFeeder("splitted_dataset/test_splitted_data")
train_feeder = utils.DataFeeder("splitted_dataset/train_splitted_data")

# Training/Loading
model = utils.ask_load("trained/lstm_enc")
if model is None:
    for _ in [0, 1]:
        for data_block in track(train_feeder.feed(), total=train_feeder.n_of_parts):

            # Flattens features
            time_series = data_block["time_series"].reshape((-1, 80, 81))
            
            global_model.fit(
                x=[data_block["toa"], time_series],
                y=data_block["outcome"],
                epochs=10,
                batch_size=128,
                shuffle=True,
                verbose=0,
            )
        global_model.save("trained/lstm_enc")
else:
    global_model = model

# Testing at the end of the super-epoch
test_block = next(test_feeder.feed())
data_block = next(train_feeder.feed())
print(f"data_block['outcome'].shape = {data_block['outcome'].shape}")
print(f"test_block['outcome'].shape = {test_block['outcome'].shape}")

print(f"data_block['toa'].shape = {data_block['toa'].shape}")
print(f"test_block['toa'].shape = {test_block['toa'].shape}")
predictions = global_model.predict([test_block['toa'], test_block['time_series'].reshape((-1, 80, 81))], verbose=0)
print(f"prediction shape is {predictions.shape}")
error = (
    np.std(
        (predictions.squeeze() - test_block["outcome"])/test_block["outcome"]
    )
    * 100
)
print(f"mean error {error:.2f}")
print("Prediction examples")
for i in range(10):
    print(
        f"true: [green]{test_block['outcome'][i]:.1f}[/] \t predicted: [blue]{predictions.squeeze()[i]:.1f}"
    )
