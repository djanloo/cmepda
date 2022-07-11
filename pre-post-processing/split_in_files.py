"""Since (issue #4) keras needs a different 'style' of generator
I decided to split each entry of the dataset in a single file.
This will probably cause a slowdown of data generation because of reading times
"""
# Make possible importing modules from parent directory
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
from cloudatlas import datafeeders, constants
from os import mkdir
from os.path import exists
from rich.progress import track

DIR = constants.DIR_DATA_BY_ENTRY

# I use the previously splitted dataset because
# loading the full one happened once and it was a miracle
test_feeder = datafeeders.DataFeeder("splitted_dataset/test_splitted_data")
train_feeder = datafeeders.DataFeeder("splitted_dataset/train_splitted_data")

# Save the datum with a specific dtype because I still don't trust indexing on dtype=object
funky_dtype = constants.funky_dtype

if not exists(DIR):
    mkdir(DIR)
    mkdir(f"{DIR}/train")
    mkdir(f"{DIR}/test")
    mkdir(f"{DIR}/validation")

# Using 70-15-15 proportions since the original dataset is 70-30

# Test and validation
c = 0
# Gets the index that splits the test dataset in half
mid_index = (len(next(test_feeder.feed())["toa"]) * test_feeder.n_of_parts) // 2
print(f"mid index is {mid_index}")
for data_block in track(test_feeder.feed(), total=test_feeder.n_of_parts):

    block_time_series = data_block["time_series"].reshape((-1, 80, 81))
    block_toas = data_block["toa"]
    block_outcomes = data_block["outcome"]

    for ts, toa, out in zip(block_time_series, block_toas, block_outcomes):

        if c < mid_index:
            path = f"{DIR}/test/part_{c}"
        else:
            path = f"{DIR}/validation/part_{c - mid_index}"
        entry = np.empty(1, dtype=funky_dtype)
        entry["toa"] = toa
        entry["time_series"] = ts
        entry["outcome"] = out
        np.save(path, entry)
        c += 1

## Training data
c = 0
for data_block in track(train_feeder.feed(), total=train_feeder.n_of_parts):

    block_time_series = data_block["time_series"].reshape((-1, 80, 81))
    block_toas = data_block["toa"]
    block_outcomes = data_block["outcome"]

    for ts, toa, out in zip(block_time_series, block_toas, block_outcomes):
        entry = np.empty(1, dtype=funky_dtype)
        entry["toa"] = toa
        entry["time_series"] = ts
        entry["outcome"] = out
        np.save(f"{DIR}/train/part_{c}", entry)
        c += 1
