"""Module for nets generation.

At the moment the best one (and the only implemented) is LstmEncoder.
"""
import numpy as np
from os.path import exists, join
import warnings

import keras
from keras.layers import LSTM, Dense, Input, Flatten, concatenate
from keras.models import Model
from keras.metrics import RootMeanSquaredError
from keras.utils.vis_utils import plot_model

from rich.progress import track
from rich import print

from datafeeders import DataFeeder
import utils


class LstmEncoder:
    """The net to analyze the AirShower dataset.

    It is composed by a `time of arrival` branch and a `time series` branch.

    The latter is designed as an encoder of dense layers. The hypothesis that brought to this
    design is the information redundancy of the time of arrival matrix. The more a paricle shower is
    homogeneous the less number of parameters are needed to describe it, such as an incidence angle,
    spread angle and height of first collision.
    The encoder aims to extract those "homogeneous beam" parameters.

    The time series branch is composed of a layer of lstm units and a relu-activated dense layer.
    It processes the time evolution of the detectors activity.

    Finally the output of the two branches are concatenated and porcessed with a small number
    of relu-activated dense layers. A final linear dense unit serves as the output.

    Since the whole net's purpose is a regression task the loss function is by default the
    mean squared error and the natural metric is the RMSE.

    Args:
        optimizer (keras.optimizers): the optimizer. By default is `Adam` with `learning_rate=0.001` .
        path (:obj:`str`, optional): the folder where the trained model is saved into.

    Attributes:
        model (keras.models.Model): the (compiled) LstmEncoder network

    """

    def __init__(self, optimizer="adam", path="trained/LstmEncoder"):

        self.optimizer = optimizer
        self.path = path

        # Time of arrival branch
        input_toa = Input(shape=(9, 9, 1), name="time_of_arrival")
        flat = Flatten()(input_toa)
        enc = Dense(9, activation="relu")(flat)
        enc = Dense(4, activation="relu")(enc)
        enc = Dense(4, activation="relu")(enc)
        encoder = Model(inputs=input_toa, outputs=enc)

        # Time series branch
        input_ts = Input(shape=(80, 81), name="time_series")
        lstm = LSTM(64)(input_ts)
        dense = Dense(16, activation="relu")(lstm)
        long_short_term_memory = Model(inputs=input_ts, outputs=dense)

        # Concatenation
        conc = concatenate([encoder.output, long_short_term_memory.output])
        z = Dense(16, activation="relu")(conc)
        z = Dense(4, activation="linear")(z)
        z = Dense(1, activation="linear")(z)

        complete_model = Model(
            inputs=[encoder.input, long_short_term_memory.input], outputs=z
        )

        complete_model.compile(
            optimizer=self.optimizer,
            loss="mean_squared_error",
            metrics=[RootMeanSquaredError()],
        )

        self.model = complete_model
        self.remote = utils.RemoteMonitor()
        self.__check_load()

    def train(self, **fit_kwargs):
        """Trains the model and saves history."""
        self.history = self.model.fit(**fit_kwargs)

        # Saves
        self.model.save(self.path)
        np.save(f"{self.path}/history", self.history)

        # Tries remote monitoring
        self.remote.send(
            [
                f"Training of {self.path} complete",
                f"Last val-loss was {self.history.history['val_loss'][-1]:.1f}",
                f"Last val-RMSE was {self.history.history['val_root_mean_squared_error'][-1]:.1f}",
            ]
        )

    def resolution_on(self, feeder):
        """Estimates the resolution on a specified dataset.

        Args:
            feeder (DataFeeder): the feeder of the dataset.
        """
        true_vals = np.array(
            [batch[1] for batch in track(feeder, description="Getting true vals ..")]
        ).reshape((-1))
        predictions = np.array(self.model.predict(test_feeder)).squeeze()

        return np.std(predictions - true_vals)

    def __check_load(self):
        # If it exists an already trained model at self.path, __check_load loads it in self.model
        if exists(self.path):
            warnings.warn(f"Trained model already present in {self.path}")
            print("Loading the model...", end=' ')
            self.model = keras.models.load_model(self.path)
            print("done!")



"""
feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

# Load the dataset feeders
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)
train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)

# Training/Loading
def train_and_resolution(path):
    model = utils.ask_load(path)
    if model is None:
        # If the model does not exist or is not loaded, execute training
        global_model = get_net()

        # Adding a scheduler for adam
        def scheduler(epoch, current_lr):
            # Splits the learning rate after epoch 5
            # Stops dividing when lr is 'too small'
            if epoch < 5:
                return current_lr
            elif current_lr < 1.0:
                return current_lr + 0.02
            else:
                return current_lr

        lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

        history = global_model.fit(
            x=train_feeder,
            epochs=25,
            validation_data=val_feeder,
            batch_size=128,
            verbose=1,
            use_multiprocessing=False,  # For some reasons multiproc doubles the training time
            callbacks=[lr_scheduler],
        )
        global_model.save(path)

        # Saves the history
        np.save(f"{path}/history", history)

        try:
            telegram_send.send(
                messages=[
                    f"Last rms was {history.history['root_mean_squared_error'][-1]:.1f}"
                ]
            )
        except:
            print("Network failed")
    else:
        # If the model is loaded shows history
        global_model = model

        # Loads the history
        if exists(join(path, "history.npy")):
            history = np.load(f"{path}/history.npy", allow_pickle=True).item()
            # print(history.history)
        plt.plot(history.history["loss"])
        plt.plot(history.history["root_mean_squared_error"])
        plt.show()

    global_model.summary()

    # I COMMITTED ON THE WRONG ISSUE
    # Test for issue #5
    # Since predictiong generator shuffles data indexes at the end #
    # What is the correct way to measure the std of (pred - true) ? #

    # Way 1: don't predict generators #
    predictions = np.array([])
    true = np.array([])
    for batch in track(test_feeder, description="getting pred-true couples .."):
        true = np.concatenate((true, batch[1]))
        predictions = np.concatenate(
            (
                predictions,
                global_model.predict(batch[0], batch_size=128, verbose=0).squeeze(),
            )
        )
    res1 = np.std(predictions - true)
    print(f"Resolution 1 is {res1}")

    print("Examples of predictions:")
    for i in range(10):
        print(f"true: [green]{true[i]:.1f}[/] \t predicted: [blue]{predictions[i]:.1f}")

    # Way 2: predict and get true #
    # THIS IS WRONG: a shuffle happens after predict, so we compare #
    # unmatching pairs pred - true #
    predictions = np.array([])
    true = np.array(
        [batch[1] for batch in track(test_feeder, description="getting true vals ..")]
    ).reshape((-1))

    predictions = np.array(global_model.predict(test_feeder)).squeeze()
    res2 = np.std(predictions - true)
    print(f"Resolution 2 is {res2}")

    print("Examples of predictions:")
    for i in range(10):
        print(f"true: [green]{true[i]:.1f}[/] \t predicted: [blue]{predictions[i]:.1f}")

    # Way 3: model.evaluate #
    # Note: the metric used is RMSE. It is different form std. #
    res3 = model.evaluate(test_feeder)[1]
    print(f"Resolution 3 is {res3}")

    return res3


if __name__ == "__main__":
    train_and_resolution("trained/albertino")
"""