import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datafeeders import FeederProf, DataFeeder
from net import LstmEncoder

from matplotlib import pyplot as plt

feeder_options = {
    "shuffle": False, # Only for testing
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

train_feeder = DataFeeder("data_by_entry_aug/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
test_feeder = DataFeeder("data_by_entry/test", **feeder_options)

claretta = LstmEncoder(path="trained/claretta")

print(claretta.resolution_on(test_feeder))

exit()

claretta.train(
    x=train_feeder,
    epochs=150,
    validation_data=val_feeder,
    batch_size=128,
    verbose=1,
    use_multiprocessing=False,
)
