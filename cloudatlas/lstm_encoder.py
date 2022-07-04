"""Test to see whether the lstm+encoder combo is a good idea

For each time series the encoder receives the same time of arrival matrix.
Perhaps this data redundance will overtrain that part of the net.

"""
import numpy as np
from os.path import exists, join

import keras
import tensorflow as tf
from keras.layers import LSTM, Dense, Input, Flatten, concatenate
from keras.models import Model
from keras.metrics import RootMeanSquaredError
from keras.utils.vis_utils import plot_model
import utils
from rich.progress import track
from rich import print
import telegram_send

# Test
import matplotlib.pyplot as plt

feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

# Load the dataset feeders
test_feeder = utils.DataFeederKeras("data_by_entry/test", **feeder_options)
train_feeder = utils.DataFeederKeras("data_by_entry/train", **feeder_options)


def get_net():
    """Used to reinitialize the model
    I am basically lazy.
    """
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

    # ATM estimation is linearly biased. Try to remove it by another linear layer
    # on what is supposed to be the estimation
    z = Dense(1, activation="relu")(z)
    z = Dense(1, activation="linear")(z)


    global_model = Model(
        inputs=[encoder.input, long_short_term_memory.input], outputs=z
    )

    global_model.compile(
        optimizer="adam",  # keras.optimizers.Adam(learning_rate=1e-4),
        loss="mean_squared_error",
        metrics=[RootMeanSquaredError()],
    )

    plot_model(global_model, show_shapes=True)
    return global_model


# Training/Loading
def train_and_resolution(path):
    model = utils.ask_load(path)
    if model is None:
        global_model = get_net()

        history = global_model.fit(
            x=train_feeder,  # fit_generator is deprecated, this can be done
            epochs=120,
            validation_data=test_feeder[0],  # Mmmh
            batch_size=128,
            verbose=1,
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
        global_model = model

        # Loads the history
        if exists(join(path, "history.npy")):
            history = np.load(f"{path}/history.npy", allow_pickle=True).item()
            # print(history.history)
        plt.plot(history.history["loss"])
        plt.plot(history.history["root_mean_squared_error"])
        plt.show()

        @tf.function
        def traceme(x):
            return model(x)

        logdir = "log"
        writer = tf.summary.create_file_writer(logdir)
        tf.summary.trace_on(graph=True, profiler=True)
        # Forward pass
        traceme([tf.zeros((1, 9, 9, 1)), tf.zeros((1, 80, 81))])
        with writer.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)

        exit()

    global_model.summary()

    predictions = np.array([])
    true = np.array([])
    for batch in track(test_feeder):
        true = np.concatenate((true, batch[1]))
        predictions = np.concatenate(
            (
                predictions,
                global_model.predict(batch[0], batch_size=128, verbose=0).squeeze(),
            )
        )
    error = np.std(predictions - true)
    print(f"mean error is {error}")
    for i in range(10):
        print(f"true: [green]{true[i]:.1f}[/] \t predicted: [blue]{predictions[i]:.1f}")
    return error


if __name__ == "__main__":
    train_and_resolution("trained/mariuccio")
