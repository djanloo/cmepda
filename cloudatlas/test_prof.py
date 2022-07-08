import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from datafeeders import FeederProf, DataFeeder
from net import LstmEncoder

feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

prof_alberto = FeederProf(
    "trained/albertino", "data_by_entry/train", difficulty_levels=5, **feeder_options
)


prof_mario = FeederProf(
    "trained/mariuccio_strong", "data_by_entry/train", difficulty_levels=5, **feeder_options
)

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

unordered_alberto_scores = np.zeros(prof_mario.scores.shape)
unordered_alberto_scores[prof_alberto.sort_order] = prof_alberto.scores

unordered_mario_scores = np.zeros(prof_mario.scores.shape)
unordered_mario_scores[prof_mario.sort_order] = prof_mario.scores

u_true = np.zeros(prof_mario._true_vals.shape)
u_true[prof_mario.sort_order]= prof_mario._true_vals

u_pred = np.zeros(prof_mario._true_vals.shape)
u_pred[prof_mario.sort_order]= prof_mario._estimates


viridis5 = cm.get_cmap("viridis", 5)
plt.scatter(u_true, u_pred/u_true,
         c=unordered_alberto_scores, cmap=viridis5,
             alpha=0.5, s=4.0)
plt.colorbar()
plt.xlim((630, 1020))
plt.ylim((0.75, 1.15))

plt.title("Data Difficulty")
plt.xlabel("true value [m]")
plt.ylabel("Nomalized prediction")
plt.show()

exit()
prof_alberto.teaching_level = 4
print(f"Prof has {len(prof_alberto)} lessons to give")

val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)
mariuccio = LstmEncoder(path="trained/mariuccio_strong")


mariuccio.train(
    x=prof_alberto,
    epochs=25,
    validation_data=val_feeder,
    batch_size=128,
    verbose=1,
    use_multiprocessing=False,
)
