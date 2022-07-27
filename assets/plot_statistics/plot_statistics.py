import matplotlib.pyplot as plt
from context import stats
from context import LstmEncoder, DataFeeder

lstmenc_aug = LstmEncoder(path="trained/lstmenc_aug")
lstmenc_vanilla = LstmEncoder(path="trained/lstmenc_train_sub")


BATCH_SIZE = 128

# Feeder options for both
feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa","time_series"],
    "target_field": "outcome",
}

# call same vanilla test dataset for both
train_feeder = DataFeeder("data_by_entry/train", **feeder_options)

stats.interpercentile_plot(
    [lstmenc_aug],
    "data_by_entry/test",
    [feeder_options],
    titles=['Augmentation']
)

plt.show()