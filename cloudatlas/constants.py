"""Module for collection of global constants"""
import numpy as np

# directory with datas splitted by entries
DIR_DATA_BY_ENTRY = "data_by_entry"
DIR_DATA_BY_ENTRY_AUG = "data_by_entry_aug"

# custom numpy dtype
funky_dtype = np.dtype(
    [
        ("outcome", np.float64),
        ("time_series", np.float32, (80, 81)),
        ("toa", np.float32, (9, 9, 1)),
    ]
)

# File format
FILENAME = "part_{name:06}.npy"
