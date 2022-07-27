import sys

from context import LstmEncoder, DataFeeder, utils, stats
from matplotlib import pyplot as  plt

BATCH_SIZE = 128
EPOCHS = 144

feeder_options = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "input_fields": ["toa","time_series"],
    "target_field": "outcome",
}
lstmenc_long_run= LstmEncoder(
    path="trained/lstmenc_long_run", 
    earlystopping=False,
    tensorboard=True,
    )

train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
val_feeder = DataFeeder("data_by_entry/validation", **feeder_options)

# lstmenc_long_run.train(
#         x=train_feeder,
#         epochs=EPOCHS,
#         validation_data=val_feeder,
#         batch_size=BATCH_SIZE,
#         verbose=1,
#         use_multiprocessing=False,
#     )

stats.interpercentile_plot(
    [lstmenc_long_run],
    "data_by_entry/test",
    [feeder_options],
    plot_type=None,
    titles=["Predictions v.s. true values"]
)

plt.show()