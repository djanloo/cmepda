from context import utils as ut
from context import DataFeeder
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"

feeder_options = {
    "shuffle": True, # Only for testing
    "batch_size": 10,
    "input_fields": ["toa", "time_series"],
    "target_field": "outcome",
}

df = DataFeeder("data_by_entry/train", **feeder_options)
ts = df[0][0][1][:].reshape((-1, 9,9))
print(ts.shape)
anim = ut.animate_time_series(ts)
plt.axis('off')
plt.title("Detector time series")
anim.save("anim.gif")