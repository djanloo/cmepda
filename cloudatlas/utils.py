"""Utility module"""
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from rich.progress import track


class DataFeeder:
    """Class that prevents memory shortage.
    
    Mainly loads the splitted dataset from the directory yielded by
    :func:`cloudatlas.utils.split_dataset`.

    Attributes
    ----------
        directory : str
            the main directory of the dataset
        n_of_parts : int
            the number of parts the dataset is made of
        axes : list
            qualitatively different sub-parts of the dataset, e.g. [input, output] or 
            [feature1, feature2, output].
        data : list
            the path for each sub-part divided by axis e.g.: 
            [[ax0_file0, ax0_file1],[ax1_file0, ax1_file1]
    """
    def __init__(self, directory):

        self.directory = directory
        self._dataset_n_of_parts = None
        self.axes = np.sort(
            [a for a in listdir(self.directory) if not isfile(join(self.directory, a))]
        )
        self.n_axes = len(self.axes)
        self.data = []
        for ax in self.axes:
            axis_directory = os.path.join(self.directory, ax)
            axis_files = [
                afile
                for afile in listdir(axis_directory)
                if isfile(join(axis_directory, afile))
            ]
            if not axis_files:
                raise FileNotFoundError(f"axis {ax} has no files!")
            axis_files = np.sort(axis_files)
            self.data.append(axis_files)
            self.n_of_parts = len(axis_files)

    @property
    def n_of_parts(self):
        return self._dataset_n_of_parts

    @n_of_parts.setter
    def n_of_parts(self, value):
        if self._dataset_n_of_parts is None:
            self._dataset_n_of_parts = value
        elif value != self._dataset_n_of_parts:
            raise ValueError("axes have different number of files")

    def feed(self):
        """Returns the generator that give the next part of the dataset using ``next()``:
        
            >>> feeder = DataFeeder("dataset")
            >>> gen = feeder.feed()
            >>> next_block = next(gen) # Gives the next block of the dataset 
        
        """
        for part in range(self.n_of_parts):
            yield tuple(
                [
                    np.load(f"{self.directory}/{axis}/{self.data[ax_index][part]}")
                    for ax_index, axis in enumerate(self.axes)
                ]
            )


def split_dataset(data, filename,axes_names, n_of_files, parent=None):
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
        raise ValueError(f"axes names and data dimension mismatch:\n"+
                         f"len(data) = {len(data)}\tlen(axes_names) = {len(axes_names)}")
    directory = ""
    if parent is not None:
        if not os.path.exists(parent):
            os.mkdir(parent)
        directory = parent
    directory = f"{directory}/{filename}_splitted_data"
    if not os.path.exists(directory):
        os.mkdir(directory)
        for ax in axes_names:
            os.mkdir(join(directory, ax))

    for axis, axis_data in track(
        zip(axes_names, list(data)), 
        description=f"saving files for '{filename}'..",
        total=n_of_files
    ):
        splitted_data = np.array_split(axis_data, n_of_files)
        for i, section in enumerate(splitted_data):
            np.save(f"{directory}/{axis}/{filename}_part_{i}", section)
    return directory

def animate_time_series(array):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    vmax, vmin = np.max(array),  np.min(array)
    fig = plt.figure()
    canvas = plt.imshow(
        np.random.uniform(0, 1, size=(9, 9)), 
                                vmin=vmin, vmax=vmax, 
                                cmap="plasma"
    )
    def animate(i):
        image = array[i]
        canvas.set_array(image)
        return (canvas,)
    return FuncAnimation(fig, animate, frames=len(array), interval=0, blit=True)