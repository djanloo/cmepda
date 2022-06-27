"""Script version of the encoder notebook"""
import numpy as np
from matplotlib import pyplot as plt
import utils

from rich.progress import track
from rich import print

# Load the dataset feeders
test_feeder = utils.DataFeeder("splitted_dataset/test_splitted_data")
train_feeder = utils.DataFeeder("splitted_dataset/train_splitted_data")

### Plot the dataset
"""
fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
fig.subplots_adjust(wspace=-0.85, hspace=-0.25)

for i, ax in enumerate(axes.flatten()):
    image = ax.imshow(next(test_feeder.feed())[0][i], cmap="plasma", vmin=-3, vmax=3)
    ax.axis('off')


cbar_ax = fig.add_axes([.85, 0.1, 0.03, 0.7])
fig.colorbar(image, cax=cbar_ax)
axes[0,1].set_title("Time of arrival on detectors")

plt.show()
"""

# Since the data manifest some sort of redundance (angle, height, dispersion angle and average speed
# give the t.o.a. matrix, 4 parameters to 81 observations)  an encoder network is tested

from keras import layers
from keras.models import Model
from keras.utils.vis_utils import plot_model

input = layers.Input(shape=(9, 9, 1))
layer = layers.LeakyReLU(alpha=0.2)(input) 
layer = layers.Dense(16, activation="relu")(layer) 
layer = layers.Dense(4, activation="relu")(layer)
layer = layers.Flatten()(layer)
layer = layers.Dense(4, activation="linear")(layer)
layer = layers.Dense(16, activation="relu")(layer) 
layer = layers.LeakyReLU(alpha=0.2)(layer) 
layer = layers.Dense(4, activation="linear")(layer)
layer = layers.Dense(1, activation="linear")(layer)

retino = Model(input, outputs=layer)
retino.compile(optimizer="adam", loss="mean_squared_error")

plot_model(retino, show_shapes=True, show_layer_names=True)
# retino.summary()

# Training
for _ in [0, 1]:
    for data_block in track(train_feeder.feed(), total=train_feeder.n_of_parts):
        retino.fit(
            x=data_block["toa"],
            y=data_block["outcome"],
            epochs=5,
            batch_size=128,
            shuffle=True,
            verbose=0,
        )

    # Testing
    test_block = next(test_feeder.feed())
    predictions = retino.predict(test_block["toa"], verbose=0).squeeze()
    error = (
        np.std(
            (predictions - test_block["outcome"])
            / test_block["outcome"]
        )
        * 100
    )
    print(f"mean error {error:.2f}")

    print("Prediction examples")
    for i in range(10):
        print(
            f"true: [green]{test_block['outcome'][i]:.1f}[/] \t predicted: [blue]{predictions[i]:.1f}"
        )
