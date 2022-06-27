"""Script version of the encoder notebook"""
import numpy as np
from matplotlib import pyplot as plt
import utils
from keras import layers
from keras.models import Model
from keras.utils.vis_utils import plot_model

from rich.progress import track
from rich import print

# Load the dataset feeders
test_feeder = utils.DataFeeder("splitted_dataset/test_splitted_data")
train_feeder = utils.DataFeeder("splitted_dataset/train_splitted_data")

input = layers.Input(shape=(9, 9, 1))
layer = layers.Dense(128, activation='relu')(input)
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Flatten()(layer)
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Dense(1, activation="linear")(layer)

retino = Model(input, outputs=layer)
retino.compile(optimizer="adam", loss="mean_squared_error")

plot_model(retino, show_shapes=True, show_layer_names=True)

# Training
for _ in [0, 1]:
    for data_block in track(train_feeder.feed(), total=train_feeder.n_of_parts):
        retino.fit(
            x=data_block["toa"],
            y=data_block["outcome"],
            epochs=2,
            batch_size=128,
            shuffle=True,
            verbose=0,
        )
    # Testing at the end of the super-epoch
    test_block = next(test_feeder.feed())
    data_block = next(train_feeder.feed())
    print(f"data_block['outcome'].shape = {data_block['outcome'].shape}")
    print(f"test_block['outcome'].shape = {test_block['outcome'].shape}")

    print(f"data_block['toa'].shape = {data_block['toa'].shape}")
    print(f"test_block['toa'].shape = {test_block['toa'].shape}")

    print(f"prediction shape is {retino.predict(test_block['toa'], verbose=0).shape}")
    error = (
        np.std(
            (retino.predict(test_block["toa"], verbose=0).squeeze() - test_block["outcome"])/test_block["outcome"]
        )
        * 100
    )
    print(f"mean error {error:.2f}")
    print("Prediction examples")
    for i in range(10):
        print(
            f"true: [green]{test_block['outcome'][i]:.1f}[/] \t predicted: [blue]{np.squeeze(retino.predict(test_block['toa'][i], verbose=0)):.1f}"
        )
