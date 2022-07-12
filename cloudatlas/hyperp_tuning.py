from keras.layers import LSTM, Dense, Input, Flatten, concatenate
from keras.models import Model
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend
import keras_tuner
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datafeeders import FeederProf, DataFeeder

# constants
EPOCHS_PATIENCE = 5  # numero di epoche dopo le quali il processo viene killato
EPOCHS = 5
BATCH_SIZE = 128

# options
feeder_options = {
    "shuffle": False,  # Only for testing
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry_aug/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)


def build_model(hp):
    # Time of arrival branch
    input_toa = Input(shape=(9, 9, 1), name="time_of_arrival")
    flat = Flatten()(input_toa)
    enc = Dense(9, activation="relu")(flat)
    enc = Dense(hp.Int("units enc dense2", min_value=4, max_value=12, step=4),
                activation=hp.Choice("activation 2", ["relu", "tanh"]))(enc)
    enc = Dense(hp.Int("units enc dense3", min_value=4, max_value=12, step=4),
                activation=hp.Choice("activation 3", ["relu", "tanh"]))(enc)
    encoder = Model(inputs=input_toa, outputs=enc)

    # Time series branch
    input_ts = Input(shape=(80, 81), name="time_series")
    lstm = LSTM(hp.Int("units LSTM", min_value=32, max_value=96, step=32))(input_ts)
    dense = Dense(hp.Int("units lstm dense", min_value=16, max_value=24, step=4),
                  activation=hp.Choice("activation lstm dense", ["relu", "tanh"]))(lstm)
    long_short_term_memory = Model(inputs=input_ts, outputs=dense)

    # Concatenation
    conc = concatenate([encoder.output, long_short_term_memory.output])
    z = Dense(hp.Int("units conc 1", min_value=16, max_value=24, step=4),
              activation=hp.Choice("activation conc", ["relu", "tanh"]))(conc)
    z = Dense(4, activation="linear")(z)
    z = Dense(1, activation="linear")(z)

    complete_model = Model(
        inputs=[encoder.input, long_short_term_memory.input], outputs=z
    )

    learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3])

    opt = Adam(learning_rate=learning_rate)

    complete_model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=[RootMeanSquaredError()],
    )

    return complete_model


# EARLY STOPPING
es = EarlyStopping(monitor="val_loss",
                   patience=EPOCHS_PATIENCE,
                   restore_best_weights=True)

# TUNER
tuner = keras_tuner.BayesianOptimization(hypermodel=build_model,
                                         objective="val_accuracy",
                                         max_trials=10,
                                         seed=42,
                                         directory="tuned/hyper_tuning",
                                         project_name="hyper_tuning_cloudatlas"
                                         )

# PERFORM HYPERPARAMETER SEARCH
tuner.search(x=train_feeder,
             epochs=EPOCHS,
             validation_data=val_feeder,
             batch_size=BATCH_SIZE,
             verbose=1,
             use_multiprocessing=False)

# print best hyperparameter
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# get the best parameters
dense_enc2 = best_hp.get("units end dense2")
act2 = best_hp.get("activation 2")
dense_enc3 = best_hp.get("units end dense3")
act3 = best_hp.get("activation 3")
lstm_units = best_hp.get("units LSTM")
dense_lstm = best_hp.get("units lstm dense")
act_lstm = best_hp.get("activation lstm dense")
conc_units = best_hp.get("units conc 1")
act_conc = best_hp.get("activation conc")
lr = best_hp.get("learning rate")


print(f"[INFO] optimal number of units in dense layer 2: {dense_enc2}")
print(f"[INFO] optimal activation dense layer 2: {act2}")
print(f"[INFO] optimal number of units in dense layer 3: {dense_enc3}")
print(f"[INFO] optimal activation dense layer 3: {act3}")
print(f"[INFO] optimal number of units in lstm layer: {lstm_units}")
print(f"[INFO] optimal number of units in dense lstm layer: {dense_lstm}")
print(f"[INFO] optimal activation dense lstm layer: {act_lstm}")
print(f"[INFO] optimal number of units in conc dense layer: {conc_units}")
print(f"[INFO] optimal activation con dense layer: {act_conc}")
print(f"[INFO] optimal for learning rate: {lr}")

