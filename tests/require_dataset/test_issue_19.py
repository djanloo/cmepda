from context import LstmEncoder, LinearProbe
from context import DataFeeder
from rich import print

BATCH_SIZE = 128
EPOCHS = 50

lstmenc = LstmEncoder(path="trained/freezing/lstmenc_train_sub")
linear_probe = LinearProbe(lstmenc)

# optionsenc
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
# test_feeder = DataFeeder("data_by_entry/test", **feeder_options)

linear_probe.train(
        x=train_feeder,
        epochs=EPOCHS,
        validation_data=val_feeder,
        batch_size=BATCH_SIZE,
        verbose=1,
        use_multiprocessing=False,
    )
