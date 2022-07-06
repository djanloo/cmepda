import numpy as np
from DataFeeders import FeederProf
import matplotlib.pyplot as plt

feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

prof_alberto = FeederProf('trained/albertino', 'data_by_entry/test', difficulty_levels=5, **feeder_options )
sc =   prof_alberto[0]['scores']
plt.hist(sc, bins=15, histtype='step', density=True)
plt.title("Difficulty distribution")
plt.xlabel("level")
plt.show()