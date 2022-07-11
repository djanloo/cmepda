"""Script to split the dataset

Has to be executed once.

Reminder: original structure is (event, 9x9 detectors, 80 ts + 1 toa)
"""
# Make possible importing modules from parent directory
import os
import sys
import numpy as np
import constants

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from pd4ml import Airshower
from os import mkdir
from rich import print
from os.path import join, exists
from rich.progress import track


THINNING = 1  # Thinning factor: takes one sample in every fixed number
perc = ()


if not exists(constants.DIR_DATA_BY_ENTRY):
    mkdir(constants.DIR_DATA_BY_ENTRY)
    mkdir(f"{constants.DIR_DATA_BY_ENTRY}/train")
    mkdir(f"{constants.DIR_DATA_BY_ENTRY}/test")
    mkdir(f"{constants.DIR_DATA_BY_ENTRY}/validation")


for mode in ["test", "train", "validation"]:
    # Load the dataset
    x, y = Airshower.load_data(mode)

    # type casting
    entries = np.empty(len(x["features"][0]), dtype=constants.funky_dtype)

    # Time of arrival is stored as a 9x9 matrix (for each event)
    entries["toa"] = x["features"][0][::THINNING, :, -1].reshape((-1, 9, 9, 1))

    # Time series is stored as a 80-array of 9x9 matrices (for each event)
    # A swap of axes is then required
    time_series = x["features"][0][::THINNING, :, :-1]
    entries["time_series"] = np.swapaxes(time_series, 1, 2).reshape((-1, 80, 9, 9))

    # right values
    entries["outcome"] = y[::THINNING]

    # Clean useless big file, free some memory
    del x

    # savings in files called part_num
    for i, entry in enumerate(entries):

        # split test in test and validation
        if mode == "test" and i > len(entries) // 2:
            folder = "validation"
            n = i - len(entries) // 2
        elif mode == "test" and i <= len(entries) // 2:
            folder = "test"
            n = i
        else:
            folder = "train"
            n = i

        np.save(f"{constants.DIR_DATA_BY_ENTRY}/{folder}/part_{n:06}.npy", entry)
