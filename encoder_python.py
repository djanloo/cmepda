"""Script version of the encoder notebook"""
import numpy as np
from matplotlib import pyplot as plt 
import utils

from rich.progress import track

# Load the dataset feeders
test_feeder = utils.data_feeder("splitted_dataset/test_splitted_data/")
train_feeder = utils.data_feeder("splitted_dataset/train_splitted_data")

### Plot the dataset

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
fig.subplots_adjust(wspace=-0.85, hspace=-0.25)

for i, ax in enumerate(axes.flatten()):
    image = ax.imshow(next(test_feeder.feed())[0][i], cmap="plasma", vmin=-3, vmax=3)
    ax.axis('off')


cbar_ax = fig.add_axes([.85, 0.1, 0.03, 0.7])
fig.colorbar(image, cax=cbar_ax)
axes[0,1].set_title("Time of arrival on detectors")

plt.show()

# Since the data manifest some sort of redundance (angle, height, dispersion angle and average speed
# give the t.o.a. matrix, 4 parameters to 81 observations)  an encoder network is tested

from keras import layers
from keras.models import Model
from keras.utils.vis_utils import plot_model

input = layers.Input(shape=(9, 9, 1))
layer = layers.Flatten()(input)
# layer = layers.Dense(81, activation="relu")(layer)
# y = layers.Dense(27, activation="relu")(layer)
# z = layers.Dense(27, activation="relu")(layer)
# minlayer = layers.Minimum()([y,z])
# maxlayer = layers.Maximum()([y,z])
# sublayer = layers.Subtract()([maxlayer, minlayer])
layer = layers.Dense(9, activation="relu")(layer) # change it to sublayer if not #
layer = layers.Dense(4, activation="relu")(layer)
layer = layers.Dense(4, activation="relu")(layer)
layer = layers.Dense(1)(layer)

retino = Model(input, outputs=layer)
retino.compile( optimizer="adam", 
                loss="mean_squared_error")

plot_model(retino, 
            show_shapes=True,
            show_layer_names=True)
# retino.summary()

for x,y in track(train_feeder.feed(), description="Training..", total=train_feeder.n_of_parts):
    retino.fit(
        x=x,
        y=y,
        epochs=50,
        batch_size=128,
        shuffle=True,
        verbose=0
    )
toa_test, y_test = next(test_feeder.feed())
errors = np.std( retino.predict(toa_test).squeeze() - y_test )/y_test*100
print(np.mean(errors))


