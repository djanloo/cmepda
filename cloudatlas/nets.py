"""Module for nets generation.

The three proposed designs are a small encoder (ToaEncoder) a time series LSTM (TimeSeriesLSTM) 
and a concatenation of the two (LstmEncoder).
"""
from os.path import exists 

import keras
from keras.layers import LSTM, Dense, Input, Flatten, concatenate, BatchNormalization
from keras.models import Model
from keras.metrics import RootMeanSquaredError
import numpy as np
from rich import print

import utils


class LushlooNet:
    """The basis class for other nets.

    It has no real model definition, just shared utility methods.

    Args:
        path (:obj:`str`): the folder where to save the model.

    Attributes:
        path (:obj:`str`): the path of the folder.
        model (:obj:`keras.models.Model`)
        tensorboard (bool, optional): specifies whether to use the TensorBoard callback
        earlystopping (bool, optional): specifies whether to use the EarlyStopping callback

    """

    def __init__(
        self, path="trained/LstmEncoder", tensorboard=False, earlystopping=False
    ):

        self.path = path

        # Since is a template for basic functions
        # this net has no real model
        self.model = None
        self.compilation_kwargs = {
            "optimizer": None,
            "loss": "mean_squared_error",
            "metrics": [RootMeanSquaredError()],
        }

        # Callbacks
        self.callbacks = []

        if tensorboard:
            ## TensorBoard callbacks
            ## Write TensorBoard logs to `./logs` directory
            self.callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=f"{self.path}/logs", histogram_freq=1
                )
            )

        if earlystopping:
            ## EarlyStopping callback
            ## By default monitor = val_loss
            self.callbacks.append(
                keras.callbacks.EarlyStopping(min_delta=0.05, patience=6)
            )

        # If no callback is used
        # sets callbacks to None
        if not self.callbacks:
            self.callbacks = None

        # Initializes remote monitoring
        self.remote = utils.RemoteMonitor()

    def train(self, **fit_kwargs):
        """Trains the model and saves history."""

        # Sets the callbacks if nothing is specified
        fit_kwargs.setdefault("callbacks", self.callbacks)

        # Fit model
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
        # feeder.shuffle = False  # Otherwise true - pred are mismatched
        print(f"Evaluating [blue]RMSE[/blue] for [yellow]{self.path}[/yellow]")
        self.eval = self.model.evaluate(x=feeder)
        print(self.eval)
        return self.eval[1]

    def _check_load(self):
        # If it exists an already trained model at self.path, __check_load loads it in self.model
        if exists(self.path):
            print(f"Trained model folder found in [yellow]{self.path}[/yellow]")
            try:
                print("Loading the model...", end=" ")
                self.model = keras.models.load_model(self.path)
            except OSError:
                print("empty model folder: model was not loaded")
                self.loaded = False
            else:
                self.loaded = True
                print("done!")
        else:
            self.loaded = False


class ToaEncoder(LushlooNet):
    """The encoder that processes time of arrival matrices.

    Args:
        path (:obj:`str`): the folder where to save the model.
        optimizer (:obj:`str` or :obj:`keras.optimizers.Optimizer`): the optimizer for the training stage. Default is Adam(lr=0.001).
        net_kwargs (optional): :class:`LushlooNet` keyword args
    """

    def __init__(self, optimizer="adam", **net_kwargs):

        super(ToaEncoder, self).__init__(**net_kwargs)

        # Sets net parameters
        self.path = net_kwargs.get("path", "trained/ToaEncoder")
        self.compilation_kwargs["optimizer"] = optimizer

        input_toa = Input(shape=(9, 9, 1), name="toa_input")
        enc = Dense(256, activation="relu", name="enc_dense_a")(input_toa)
        enc = Dense(128, activation="relu", name="enc_dense_b")(enc)
        enc = Dense(64, activation="relu", name="enc_dense_c")(enc)
        enc = Dense(1, activation="relu", name="enc_dense_d")(enc)
        flat = Flatten(name="enc_flatten")(enc)

        # Now adds an output layer
        # This will be removed when used in LstmEncoder
        enc = Dense(1, activation="linear", name="enc_out")(flat)

        self.model = Model(inputs=input_toa, outputs=enc, name="ToAEncoder")
        self.model.compile(**self.compilation_kwargs)
        self.model.summary()
        self._check_load()


class TimeSeriesLSTM(LushlooNet):
    """The lstm net that processes time series matrices.

    Args:
        optimizer (:obj:`str` or :obj:`keras.optimizers.Optimizer`): the optimizer for the training stage. Default is Adam(lr=0.001).
        net_kwargs (optional): :class:`LushlooNet` keyword args
    """

    def __init__(self, optimizer="adam", **net_kwargs):

        super(TimeSeriesLSTM, self).__init__(**net_kwargs)

        self.path = net_kwargs.get("path", "trained/TimeSeriesLSTM")
        self.compilation_kwargs["optimizer"] = optimizer

        input_ts = Input(shape=(80, 81), name="ts_input")
        lstm = LSTM(64, name="lstm_lstm")(input_ts)
        # dense = Dense(64, activation="relu", name="lstm_dense")(lstm)
        dense = Dense(16, activation="relu", name="lstm_dense")(lstm)
        # Now adds an output layer
        # This will be removed when used in LstmEncoder
        dense = Dense(1, activation="linear", name="lstm_out")(dense)

        self.model = Model(inputs=input_ts, outputs=dense, name="TSLstm")
        self.model.compile(**self.compilation_kwargs)
        self._check_load()


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
        optimizer (:obj:`str` or :obj:`keras.optimizers.Optimizer`): the optimizer for the training stage. Default is Adam(lr=0.001).
        net_kwargs (optional): :class:`LushlooNet` keyword args
        encoder (:obj:`keras.models.Model`, optional): the encoder sub-network. If nothing is given, a new empty encoder
            is created.
        lstm (:obj:`keras.models.Model`): the lstm sub-network. If nothing is given, a new empty lstm
            is created.
        train_encoder (bool, optional): specify whether to train or not the encoder, in case a pre trained one is given.
        train_lstm (bool, optional): specify whether to train or not the lstm, in case a pre trained one is given.

    Attributes:
        model (keras.models.Model): the (compiled) LstmEncoder network

    """

    def __init__(
        self,
        optimizer="adam",
        encoder=None,
        lstm=None,
        train_encoder=True,
        train_lstm=True,
        train_whole=True,
        **net_kwargs,
    ):

        super(LstmEncoder, self).__init__(**net_kwargs)

        self.compilation_kwargs["optimizer"] = optimizer
        self.path = net_kwargs.get("path", "trained/LstmEncoder")

        # Time of arrival branch
        self.encoder = ToaEncoder() if encoder is None else encoder
        self.encoder.model.trainable = train_encoder if encoder is not None else True

        # Time series branch
        self.lstm = TimeSeriesLSTM() if lstm is None else lstm
        self.lstm.model.trainable = train_lstm if lstm is not None else True

        # Concatenation:
        # Takes the second-last layer of the net
        conc = concatenate(
            [self.encoder.model.layers[-2].output, self.lstm.model.layers[-2].output],
            name="concatenate",
        )
        z = Dense(16, activation="relu", name="lstmenc_dense_1")(conc)
        z = Dense(16, activation="relu", name="lstmenc_dense_2")(z)
        z = Dense(16, activation="relu", name="lstmenc_dense_3")(z)
        z = Dense(1, activation="linear", name="lstmenc_out")(z)

        self.model = Model(
            inputs=[self.encoder.model.input, self.lstm.model.input],
            outputs=z,
            name="LstmEncoder",
        )

        self.model.compile(**self.compilation_kwargs)
        self.model.summary()
        self._check_load()

        # Set the net as untrainable only if it was loaded
        if self.loaded:
            self.model.trainable = train_whole

class LinearProbe(LushlooNet):

    def __init__(self, lstmencoder, **net_kwargs):

        super(LinearProbe, self).__init__(**net_kwargs)

        # Set as default adam
        self.compilation_kwargs["optimizer"] = "adam"
        
        self.path = net_kwargs.get("path", "trained/LinearProbe")

        self.lstmencoder = lstmencoder 
        self.lstmencoder.model.trainable = False 
        linear_probe_input = self.lstmencoder.model.get_layer(name="lstmenc_out").output
        bb = BatchNormalization()(linear_probe_input)
        w = Dense(4, activation="relu")(bb)
        w = Dense(4, activation="relu")(w)
        w = Dense(4, activation="relu")(w)
        w = Dense(4, activation="relu")(w)
        w = Dense(1)(w)

        self.model = Model(
                inputs=self.lstmencoder.model.input,
                outputs=w,
                name="LinearProbe",
                )
        print(f"[green]Net built correctly[/green]")

        self.model.compile(**self.compilation_kwargs)
        self.model.summary()
