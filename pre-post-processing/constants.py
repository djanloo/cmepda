"""Module for collection of global constants"""
# directory with datas splitted by entries
DIR_DATA_BY_ENTRY = "data_by_entry"

# custom numpy dtype
funky_dtype = np.dtype(
    [
        ("outcome", np.float64),
        ("time_series", np.float32, (80, 81)),
        ("toa", np.float32, (9, 9, 1)),
    ]
)