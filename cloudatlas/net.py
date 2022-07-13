"""Module for nets generation.

At the moment the best one (and the only implemented) is LstmEncoder.
"""
from re import S
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

import utils


class LushlooNet:
    """The basis class for other nets. 
    
    It has no real model definition, just utility methods.
    """
    def __init_(self, path="trained/LstmEncoder"):

        self.path = path

        # Since is a template for basic functions
        # this net has no real model
        self.model = None

        self.remote = utils.RemoteMonitor()

    def train(self, **fit_kwargs):
        """Trains the model and saves history."""
        self.history = self.model.fit(**fit_kwargs)

        # Saves
        print("Saving..")
        self.model.save(self.path)
        print(f"Saved to {self.path}")
        np.save(f"{self.path}/history", self.history)

        # Tries remote monitoring
        try:
            self.remote.send(
                [
                    f"Training of {self.path} complete",
                    f"Last val-loss was {self.history.history['val_loss'][-1]:.1f}",
                    f"Last val-RMSE was {self.history.history['val_root_mean_squared_error'][-1]:.1f}",
                ]
            )
        except Exception as e:
            print(f"Error occurred while remote monitoring: {e}")

    def resolution_on(self, feeder):
        """Estimates the resolution on a specified dataset.

        Args:
            feeder (DataFeeder): the feeder of the dataset.
        """
        feeder.shuffle = False  # Otherwise true - pred are mismatched
        true_vals = np.array(
            [batch[1] for batch in track(feeder, description="Getting true vals ..")]
        ).reshape((-1))
        predictions = np.array(self.model.predict(feeder)).squeeze()

        return np.std(predictions - true_vals)

    def __check_load(self):
        # If it exists an already trained model at self.path, __check_load loads it in self.model
        if exists(self.path):
            print(f"Trained model already present in [yellow]{self.path}[/yellow]")
            print("Loading the model...", end=" ")
            self.model = keras.models.load_model(self.path)
            print("done!")


class ToaEncoder(LushlooNet):
    """The encoder that processes time of arrival matrices."""
    def __init__(self, path="train/ToaEncoder", optimizer="adam"):

        super().__init__(path=path, optimizer=optimizer)

        # Sets net parameters
        self.path = path
        self.optimizer = optimizer

        input_toa = Input(shape=(9, 9, 1), name="time_of_arrival")
        flat = Flatten()(input_toa)
        enc = Dense(9, activation="relu")(flat)
        enc = Dense(4, activation="relu")(enc)
        enc = Dense(4, activation="relu")(enc)
        self.model = Model(inputs=input_toa, outputs=enc)
        self.__check_load()


class TimeSeriesLSTM(LushlooNet):
    """The lstm net that processes time series matrices."""
    def __init__(self, path="trained/TimeSeriesLSTM", optimizer="adam"):

        super().__init__(path=path, optimizer=optimizer)

        self.path = path
        self.optimizer = optimizer

        input_ts = Input(shape=(80, 81), name="time_series")
        lstm = LSTM(64)(input_ts)
        dense = Dense(16, activation="relu")(lstm)
        self.model = Model(inputs=input_ts, outputs=dense)
        self.__check_load()


class LstmEncoder(LushlooNet):
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
        encoder = ToaEncoder()

        # Time series branch
        lstm = TimeSeriesLSTM()

        # Concatenation
        conc = concatenate([encoder.model.output, lstm.model.output])
        z = Dense(16, activation="relu")(conc)
        z = Dense(4, activation="linear")(z)
        z = Dense(1, activation="linear")(z)

        self.model = Model(inputs=[encoder.model.input, lstm.model.input], outputs=z)

        self.model.compile(
            optimizer=self.optimizer,
            loss="mean_squared_error",
            metrics=[RootMeanSquaredError()],
        )

        self.__check_load()
