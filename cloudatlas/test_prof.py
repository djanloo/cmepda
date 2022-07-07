import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from DataFeeders import FeederProf
import matplotlib.pyplot as plt

from matplotlib import cm

feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

prof_alberto = FeederProf(
    "trained/albertino", "data_by_entry/test", difficulty_levels=5, **feeder_options
)
viridis5 = cm.get_cmap("viridis", 5)
plt.scatter(prof_alberto._true_vals, prof_alberto._estimates/prof_alberto._true_vals,
         c=prof_alberto.scores, cmap=viridis5,
             alpha=0.5, s=4.0)
plt.colorbar()
plt.xlim((630, 1020))
plt.ylim((0.75, 1.15))

plt.title("Data Difficulty")
plt.xlabel("true value [m]")
plt.ylabel("Nomalized prediction")
plt.show()
