"""Script to split and augment the dataset

Has to be executed once.

Reminder: original structure is (event, 9x9 detectors, 80 ts + 1 toa)
"""
from os import mkdir
from os.path import exists
import os

import numpy as np
from rich.progress import track


from pd4ml import Airshower
from context import constants
from context import Augment


THINNING = 1  # Thinning factor: takes one sample in every fixed number
perc = [0.7, 0.2]  # percentage where to cut for train, test, validation data
np.random.seed(42)  # seed (try1: 42 ,try2: 666, try3: 75, tr4: 3112, try5: 1997)


def save_by_line(array, directory):
    """Save a part of the dataset line by line in files"""
    for index, record in track(
        enumerate(array), description=f"Saving {directory}", total=len(array)
    ):
        fname = constants.FILENAME.format(name=index)
        np.save(f"{directory}/{fname}", record)

def load_by_line(directory):
    """Loads files and put them in entries"""
    len_dir = len(os.listdir(directory))
    entries_loaded = np.empty(len_dir, dtype=constants.funky_dtype)

    for index, fname in track(enumerate(os.listdir(directory)),
                              description=f'Loading dataset from {directory}',
                              total=len(os.listdir(directory))):
        entries_loaded[index] = np.load(f"{directory}/{fname}")

    return entries_loaded

# check on directories
if not exists(constants.DIR_DATA_BY_ENTRY_HEIGHT):
    mkdir(constants.DIR_DATA_BY_ENTRY_HEIGHT)

if not exists("dataset_presplit"):
    mkdir("dataset_presplit")

# type casting
entries = np.empty(0, dtype=constants.funky_dtype)

for mode in ["test", "train"]:
    # Load the dataset
    print(f"[INFO] About to load Airshower mode:{mode}...", end='')
    x, y = Airshower.load_data(mode)
    print("done!")

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

# temporary saving
save_by_line(entries, "dataset_presplit")

# initialize and run augmentation
aug = Augment(dataset_dir="dataset_presplit", height_threshold=850)
aug.augment_dataset()

# load augmented entries
entries_aug = load_by_line("dataset_presplit")

# prepare directories of train, test, validation
directories = ["train", "test", "validation"]
for dir_name in directories:
    if not exists(f"{constants.DIR_DATA_BY_ENTRY_HEIGHT}/{dir_name}"):
        mkdir(f"{constants.DIR_DATA_BY_ENTRY_HEIGHT}/{dir_name}")

# index where I got to split
perc = np.cumsum(np.array(perc) * len(entries_aug)).astype(int)

# 3MENDO no more
train, test, validation = np.split(entries_aug, perc)

# savings in files called part_num
for arr, dir_name in zip([train, test, validation], ["train", "test", "validation"]):
    save_by_line(arr, os.path.join(constants.DIR_DATA_BY_ENTRY_HEIGHT,dir_name))
