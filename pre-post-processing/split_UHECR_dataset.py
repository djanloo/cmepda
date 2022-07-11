"""Script to split the dataset

Has to be executed once.

Reminder: original structure is (event, 9x9 detectors, 80 ts + 1 toa)
"""
# Make possible importing modules from parent directory
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from pd4ml import Airshower
from os.path import join
import cloudatlas.utils as utils
import numpy as np
from rich import print

THINNING = 1  # Thinning factor: takes one sample in every fixed number


def split_dataset(data, filename, axes_names, n_of_files, parent=None):
    """Splits the dataset in smaller parts.
    Args
    ----
        data : tuple
            the data to be splitted, given in the following format:
                (array1, array2, array3, ... )
        filename : str
            the root of the files' names
        axes_names : list of str
            the list of names for the axes
        n_of_files : int
            the number of parts
        parent : str, optional
            the folder where the splitted dataset is placed into.
    """
    if len(axes_names) != len(data):
        raise ValueError(
            f"axes names and data dimension mismatch:\n"
            + f"len(data) = {len(data)}\tlen(axes_names) = {len(axes_names)}"
        )
    directory = ""
    if parent is not None:
        if not exists(parent):
            os.mkdir(parent)
        directory = parent
    directory = f"{directory}/{filename}_splitted_data"
    if not exists(directory):
        os.mkdir(directory)
        for ax in axes_names:
            os.mkdir(join(directory, ax))

    for axis, axis_data in track(
        zip(axes_names, list(data)),
        description=f"saving files for '{filename}'..",
        total=n_of_files,
    ):
        splitted_data = np.array_split(axis_data, n_of_files)
        for i, section in enumerate(splitted_data):
            np.save(f"{directory}/{axis}/{filename}_part_{i}", section)

    # Make config file
    splitConf.FromNames(directory, axes_names)
    return directory


for mode in ["test", "train"]:
    # Load the dataset
    x, y = Airshower.load_data(mode)
    print(f"dataset shape is {x['features'][0].shape}")

    # Time of arrival is stored as a 9x9 matrix (for each event)
    toa = x["features"][0][::THINNING, :, -1].reshape((-1, 9, 9, 1))
    print(f"toa has dtype={toa.dtype}")

    # Time series is stored as a 80-array of 9x9 matrices (for each event)
    # A swap of axes is then required
    time_series = x["features"][0][::THINNING, :, :-1]
    time_series = np.swapaxes(time_series, 1, 2).reshape((-1, 80, 9, 9))
    print(f"time_series has dtype={time_series.dtype}")

    y = y[::THINNING]
    print(f"outcome has dtype={y.dtype}")

    # Clean useless big file
    del x

    test_directory = split_dataset(
        (toa, time_series, y),
        mode,
        ["toa", "time_series", "outcome"],
        100,  # Number of parts
        parent="splitted_dataset",
    )
