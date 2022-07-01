"""Test to see whether the lstm+encoder combo is a good idea

For each time series the encoder receives the same time of arrival matrix.
Perhaps this data redundance will overtrain that part of the net.

"""
import numpy as np

from keras.layers import LSTM, Dense, Input, Flatten, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model

import utils
from rich.progress import track
from rich import print
import telegram_send


feeder_options = {  "batch_size":128, 
                    "input_fields": ["toa", "time_series"], 
                    "target_field": "outcome"
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
    input_ts = Input(shape=(80,81), name="time_series" )
    lstm = LSTM(64)(input_ts)
    dense = Dense(9, activation='relu')(lstm)
    long_short_term_memory = Model(inputs=input_ts, outputs=dense)

    # Concatenation
    conc = concatenate([encoder.output, long_short_term_memory.output])
    z = Dense(16, activation="relu")(conc)
    z = Dense(4, activation="linear")(z)
    z = Dense(1, activation="linear")(z)
    global_model = Model(inputs=[encoder.input, long_short_term_memory.input], outputs=z)

    global_model.compile(optimizer="adam", loss="mean_squared_error")
    # plot_model(global_model, show_shapes=True)
    return global_model



# Training/Loading
def train_and_resolution(path):
    model = utils.ask_load(path)
    if model is None:
        global_model = get_net()
        
        history = global_model.fit(
            x=train_feeder, # fit_generator is deprecated, this can be done
            epochs=3, # This must be so because each block has 300 events (?)
            batch_size=128,
            verbose=1
            )
        global_model.save(path)
        try:
            telegram_send.send(messages=[f"Last sqr loss was {np.sqrt(history.history['loss'][-1])}"])
        except:
            print("Network failed")
    else:
        global_model = model

    global_model.summary()

    predictions = np.array([])
    true = np.array([])
    for test_block in track(test_feeder.feed(), 
                            total=test_feeder.n_of_parts):
        true = np.concatenate((true, test_block['outcome']))
        predictions = np.concatenate((predictions,
                                    global_model.predict([ test_block['toa'], 
                                                            test_block['time_series'].reshape((-1, 80, 81))], 
                                                            verbose=0).squeeze()))
    error = np.std(predictions - true)
    print(f"mean error is {error}")
    for i in range(10):
        print(
            f"true: [green]{true[i]:.1f}[/] \t predicted: [blue]{predictions[i]:.1f}"
        )
    return error

if __name__ == "__main__":
    train_and_resolution('trained/lstm_enc')
    
