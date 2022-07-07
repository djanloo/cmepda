import numpy as np
from DataFeeders import FeederProf
import matplotlib.pyplot as plt

feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

prof_alberto = FeederProf('trained/albertino', 'data_by_entry/test', difficulty_levels=5, **feeder_options )
prof_alberto.learning_level = 1
print(len(prof_alberto))

plt.plot(prof_alberto.scores)
plt.show()


