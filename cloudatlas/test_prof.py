import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from datafeeders import FeederProf, DataFeeder
from net import LstmEncoder

from matplotlib import pyplot as plt

feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

prof_alberto = FeederProf(
    "trained/albertino", "data_by_entry/train", difficulty_levels=5, n_of_epochs=25, **feeder_options
)

val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
claretta = LstmEncoder(path="trained/claretta")

# for i in range(5):
#     data = prof_alberto[i]
#     print(f"Len of alberto is {len(prof_alberto)}")
#     idx = prof_alberto.last_batch_indexes
#     norm_est = prof_alberto._estimates[idx]/prof_alberto._true_vals[idx]
#     plt.scatter(prof_alberto._true_vals[idx],norm_est, zorder=100-i, label=f"epoch {i}")
#     prof_alberto.on_epoch_end()
# plt.legend()
# plt.show()

claretta.train(
    x=prof_alberto,
    epochs=25,
    # validation_data=val_feeder,
    batch_size=128,
    verbose=0,
    use_multiprocessing=False,
)
