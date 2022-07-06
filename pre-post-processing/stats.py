# Make possible importing modules from parent directory
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import matplotlib.pyplot as plt
from cloudatlas import utils
from cloudatlas.utils import DataFeeder
from rich.progress import track
from scipy.stats import gaussian_kde

from matplotlib import rcParams, cm

rcParams["font.family"] = "serif"

from rich.progress import track
from rich import print

FILE = "true_vs_predictions.npy"
if not os.path.exists(FILE):
    model = utils.ask_load("trained/albertino")
    if model is None:
        exit("Dumb")

    test_feeder = DataFeeder("splitted_dataset/test_splitted_data")

    predictions = np.array([])
    true_vals = np.array([])

    for block in track(test_feeder.feed(), total=test_feeder.n_of_parts):
        predictions = np.concatenate(
            (
                predictions,
                model.predict(
                    [block["toa"], block["time_series"].reshape((-1, 80, 81))],
                    verbose=0,
                    batch_size=128
                ).squeeze(),
            )
        )
        true_vals = np.concatenate((true_vals, block["outcome"]))

    np.save(FILE, np.stack((true_vals, predictions)))

else:
    print(f"Data loaded from [red]{FILE}[/red]. Delete it if the net is new.")
    true_vals, predictions = np.load(FILE)


## Quantile band
print("Quantile band started")
plt.figure(3)
predictions /= true_vals
plt.scatter(true_vals, predictions, s=0.5, alpha=0.1, color="k")

N = 30
vals = np.linspace(650, 1000, N)
delta_quants = [80, 50, 25, 10]
ups = np.zeros((len(delta_quants), N))
downs = np.zeros((len(delta_quants), N))
colormap = cm.get_cmap("plasma")

# Interval percentile estimation
for i, delta_quant in enumerate(delta_quants):
    for segment in track(range(N - 1), description=f"{delta_quant}-interpercentile:"):
        mask = (true_vals >= vals[segment]) & (true_vals < vals[segment + 1])
        down, up = np.percentile(
            predictions[mask], [50 - delta_quant / 2, 50 + delta_quant / 2]
        )
        m = np.mean(predictions[mask])
        ups[i, segment] = up
        downs[i, segment] = down

# Fill each band
for i in range(len(delta_quants) - 1):
    plt.fill_between(
        vals[:-1],
        downs[i, :-1],
        downs[i + 1, :-1],
        color=colormap(1 - delta_quants[i] / 100.0),
        alpha=0.5,
        label=f"{delta_quants[i]}",
    )
    plt.fill_between(
        vals[:-1],
        ups[i, :-1],
        ups[i + 1, :-1],
        color=colormap(1 - delta_quants[i] / 100.0),
        alpha=0.5,
    )
plt.fill_between(
    vals[:-1],
    ups[-1, :-1],
    downs[-1, :-1],
    label=f"{delta_quants[-1]}",
    color=colormap(1 - delta_quants[-1] / 100.0),
    alpha=0.5,
)
plt.legend()

plt.xlabel("true height [m]")
plt.ylabel("Normalized predictions [a.u.]")
plt.title("Heteroskedastic interpercentile ranges")

plt.xlim(640, 1010)
plt.ylim(0.75, 1.15)

plt.show()
exit()
## KDE
data = np.stack((predictions, true_vals), axis=0)
kern = gaussian_kde(data)

x = np.linspace(650, 850, 200)
y = x.copy()

X, Y = np.meshgrid(x, y)
u = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kern(u).T, X.shape)
plt.contourf(X, Y, Z)
plt.title("true vs estimated density of points")
plt.xlabel("true")
plt.ylabel("estimated")

## Plot the conditional density
## Since for a given datum omega=(time_of_arrival, time_series) correspond to two values
## z_true(omega) and z_pred(omega)
## plotting all z_true and z_pred gives the density distribution p(z_true, z_pred)
## however we want to visualize the effectiveness of the prediction neglecting the distribution of
## z_true (we must thus renormalize the number of predictions p(z_pred)d_omega with the number of true values
## lying in d_omega = p(z_true)d_omega)
## What we obtain is p(z_true, z_pred)/p(z_true) = p(z_pred | z_true) that is the conditional
## distribution, answering to the question "How are the predicted values approximating z_true distributed for a given z_true?"
plt.figure(2)
true_kern = gaussian_kde(true_vals)
Z_true = true_kern(x)
Z_true = np.tile(Z_true, len(x)).reshape((-1,) + Z_true.shape)
plt.contourf(X, Y, Z / Z_true)
plt.title("Conditional density")
plt.xlabel("true")
plt.ylabel("estimated")
plt.show()
