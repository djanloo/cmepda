"""Utility module"""
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from rich.progress import track 

class data_feeder:

    def __init__(self, directory):

        self.directory = directory
        self._dataset_n_of_parts = None
        self.axes = np.sort([a for a in listdir(self.directory) if not isfile(join(self.directory, a))])
        self.n_axes = len(self.axes)
        self.data = []
        for ax in self.axes:
            axis_directory = os.path.join(self.directory,ax)
            axis_files =  [afile for afile in listdir(axis_directory) if isfile(join(axis_directory, afile))]
            if not axis_files:
                raise FileNotFoundError(f"axis {ax} has no files!")
            axis_files = np.sort(axis_files)
            self.data.append(axis_files)
            self.dataset_n_of_parts = len(axis_files)

    @property
    def dataset_n_of_parts(self):
        return self._dataset_n_of_parts

    @dataset_n_of_parts.setter
    def dataset_n_of_parts(self, value):
        if self.dataset_n_of_parts is None:
            self._dataset_n_of_parts = value
        elif value != self._dataset_n_of_parts:
            raise ValueError("axes have different number of files")
    
    def feed(self):
        for part in range(self.dataset_n_of_parts):
            yield tuple([np.load(f"{self.directory}/{axis}/{self.data[ax_index][part]}") for ax_index, axis in enumerate(self.axes)])

def split_dataset(data, filename, n_of_files, parent=None):

    directory = ""
    if parent is not None:
        if not os.path.exists(parent):
            os.mkdir(parent)
        directory = parent
    directory = f"{directory}/{filename}_splitted_data"
    if not os.path.exists(directory):
        os.mkdir(directory)
        os.mkdir(join(directory,"x"))
        os.mkdir(join(directory,"y"))

    for axis, axis_data in track(zip(["x", "y"], list(data)), description=f"saving files for '{filename}'.."):
        splitted_data = np.array_split(axis_data, n_of_files)
        for i, section in enumerate(splitted_data):
            np.save(f"{directory}/{axis}/{filename}_part_{i}", section)
    return directory