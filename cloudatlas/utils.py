"""Utility module"""
import numpy as np
import os
from os.path import join, exists
from rich.progress import track
from rich import print
from keras.models import load_model

# Turn off keras warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def ask_load(path):
    """Conditionally loads a saved Model.

    Arguments
    ---------
        path : str
            the path of the Model
    """
    if os.path.exists(path):
        print(
            f"Existing model found at [green]{path}[/green]. Do you want to load it? [blue](y/n)"
        )
        ans = input()
        if ans == "y":
            return load_model(path)
        elif ans == "n":
            return None
        else:
            return ask_load(path)
    else:
        return None


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


def animate_time_series(array):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    vmax, vmin = np.max(array), np.min(array)
    fig = plt.figure()
    canvas = plt.imshow(
        np.random.uniform(0, 1, size=(9, 9)), vmin=vmin, vmax=vmax, cmap="plasma"
    )

    def animate(i):
        image = array[i]
        canvas.set_array(image)
        return (canvas,)

    return FuncAnimation(fig, animate, frames=len(array), interval=0, blit=True)
