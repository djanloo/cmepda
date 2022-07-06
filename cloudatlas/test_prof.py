import numpy as np
from utils import FeederProf

feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

prof_alberto = FeederProf('trained/albertino', 'data_by_entry/test', **feeder_options )
print(prof_alberto[0])