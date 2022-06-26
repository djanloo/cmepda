"""Script to split the dataset

Has to be executed once.

Reminder: original structure is (event, 9x9 detectors, 80 ts + 1 toa)
"""
from pd4ml import Airshower
from matplotlib import pyplot as plt
import os
from os.path import join
import cloudatlas.utils as utils
import numpy as np

THINNING = 1  # Thinning factor: takes one sample in every fixed number

for mode in ["test", "train"]:
    # Load the dataset
    x, y = Airshower.load_data(mode)
    print(f"dataset shape is {x['features'][0].shape}")

    # Time of arrival is stored as a 9x9 matrix (for each event)
    toa_test = x["features"][0][::THINNING, :, -1].reshape((-1, 9, 9, 1))

    # Time series is stored as a 80-array of 9x9 matrices (for each event)
    # A swap of axes is then required
    time_series_test = x["features"][0][::THINNING, :, :-1]
    time_series_test = np.swapaxes(time_series_test, 1,2).reshape((-1, 80, 9, 9))

    y = y[::THINNING]

    # Clean useless big file
    del x

    test_directory = utils.split_dataset(
        (toa_test, time_series_test, y), mode,
        ["toa", "time_series", "outcome"],
        100, # Number of parts
        parent="splitted_dataset"
    )
