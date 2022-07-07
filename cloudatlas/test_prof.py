import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from DataFeeders import FeederProf, DataFeederKeras
from lstm_encoder import get_net

import telegram_send

feeder_options = {
    "batch_size": 128,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

prof_alberto = FeederProf(
    "trained/albertino", "data_by_entry/train", difficulty_levels=5, **feeder_options
)
val_feeder = DataFeederKeras("data_by_entry/validation", **feeder_options)

mariuccio = get_net()
PATH = "trained/mariuccio"
history = mariuccio.fit(
            x=prof_alberto,
            epochs=300,
            validation_data=val_feeder,
            batch_size=128,
            verbose=1,
            use_multiprocessing=False,  # For some reasons multiproc doubles the training time
        )

# Saves net
mariuccio.save(PATH)
# Saves the history
np.save(f"{PATH}/history", history)

try:
    telegram_send.send(
        messages=[
            f"Last rms was {history.history['root_mean_squared_error'][-1]:.1f}"
        ]
    )
except:
    print("Network failed")