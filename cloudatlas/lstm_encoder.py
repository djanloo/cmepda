"""Test to see whether the lstm+encoder combo is a good idea"""
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import LSTM, Dense, Input, Flatten, concatenate
from keras.models import Model


input_toa = Input(shape=(9, 9, 1))
flat = Flatten()(input)
enc = Dense(9, activation="relu")(flat)  
enc = Dense(4, activation="relu")(enc)
enc = Dense(4, activation="relu")(enc)
encoder = Model(inputs=input_toa, outputs=enc)

input_ts = Input(shape=(9,9,80))
lstm = LSTM(8)(input_ts)
lstm = Dense(4)(lstm)
long_short_term_memory = Model(inputs=input_ts, outputs=lstm)

conc = concatenate([encoder.output, long_short_term_memory.output])
z = Dense(4, activation="relu")(conc)
z = Dense(1, activation="linear")(z)
global_model = Model(inputs=[encoder.input, long_short_term_memory.input], outputs=z)

global_model.compile(optimizer="adam", loss="mean_squared_error")

