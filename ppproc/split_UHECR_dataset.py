"""Script to split the dataset

Has to be executed once.

Reminder: original structure is (event, 9x9 detectors, 80 ts + 1 toa)
"""
from os import mkdir
from os.path import exists

import numpy as np
from rich.progress import track


from pd4ml import Airshower
from context import constants


THINNING = 1  # Thinning factor: takes one sample in every fixed number
perc = [0.7, 0.2]  # percentage where to cut for train, test, validation data
np.random.seed(42)  # seed (try1: 42 ,try2: 666, try3: 75, tr4: 3112, try5: 1997)


def save_by_line(array, directory):
    """Save a part of the dataset line by line in files"""
    for index, record in track(
        enumerate(array), description=f"Saving {directory}", total=len(array)
    ):
        fname = constants.FILENAME.format(name=index)
        np.save(f"{constants.DIR_DATA_BY_ENTRY}/{directory}/{fname}", record)


if not exists(constants.DIR_DATA_BY_ENTRY):
    mkdir(constants.DIR_DATA_BY_ENTRY)

directories = ["train", "test", "validation"]
for dir_name in directories:
    if not exists(f"{constants.DIR_DATA_BY_ENTRY}/{dir_name}"):
        mkdir(f"{constants.DIR_DATA_BY_ENTRY}/{dir_name}")

# type casting
entries = np.empty(0, dtype=constants.funky_dtype)

for mode in ["test", "train"]:
    # Load the dataset
    x, y = Airshower.load_data(mode)

    # dummy
    dummy = np.empty(len(x["features"][0]), dtype=constants.funky_dtype)

    # Time of arrival is stored as a 9x9 matrix (for each event)
    dummy["toa"] = x["features"][0][::THINNING, :, -1].reshape((-1, 9, 9, 1))

    # Time series is stored as a 80-array of 9x9 matrices (for each event)
    # A swap of axes is then required
    time_series = x["features"][0][::THINNING, :, :-1]
    dummy["time_series"] = np.swapaxes(time_series, 1, 2).reshape((-1, 80, 81))

    # right values
    dummy["outcome"] = y[::THINNING]

    # appendino
    entries = np.append(entries, dummy)

# shuffle
np.random.shuffle(entries)


# index where I got to split
perc = np.cumsum(np.array(perc) * len(entries)).astype(int)

# 3MENDO no more
train, test, validation = np.split(entries, perc)

# savings in files called part_num
for arr, dir_name in zip([train, test, validation], ["train", "test", "validation"]):
    save_by_line(arr, dir_name)
