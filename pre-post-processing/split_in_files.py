"""Since (issue #4) keras needs a different 'style' of generator
I decided to split each entry of the dataset in a single file.
This will probably cause a slowdown of data generation because of reading times
"""
# Make possible importing modules from parent directory
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
from cloudatlas import utils
from os import mkdir
from os.path import exists, join
from rich.progress import track

DIR = "data_by_entry"

## I use the previously splitted dataset because
# loading the full one happened once and it was a miracle
test_feeder = utils.DataFeeder("splitted_dataset/test_splitted_data")
train_feeder = utils.DataFeeder("splitted_dataset/train_splitted_data")

# Save the datum with a specific dtype because I still don't trust indexing on dtype=object
funky_dtype = np.dtype(
    [
        ("outcome", np.float64),
        ("time_series", np.float32, (80, 81)),
        ("toa", np.float32, (9, 9, 1)),
    ]
)

if not exists(DIR):
    mkdir(DIR)
    mkdir(f"{DIR}/train")
    mkdir(f"{DIR}/test")

# Brutally use a counter because i'm lazy
for mode, feeder in zip(["test", "train"], [test_feeder, train_feeder]):
    c = 0
    for data_block in track(feeder.feed(), total=feeder.n_of_parts):

        block_time_series = data_block["time_series"].reshape((-1, 80, 81))
        block_toas = data_block["toa"]
        block_outcomes = data_block["outcome"]

        for ts, toa, out in zip(block_time_series, block_toas, block_outcomes):
            entry = np.empty(1, dtype=funky_dtype)
            entry["toa"] = toa
            entry["time_series"] = ts
            entry["outcome"] = out
            np.save(f"{DIR}/{mode}/part_{c}", entry)
            c += 1
