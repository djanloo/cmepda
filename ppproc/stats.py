# Make possible importing modules from parent directory
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import rcParams, cm
from rich import print
from rich.progress import track

from context import LstmEncoder, ToaEncoder
from context import utils
from context import DataFeeder


rcParams["font.family"] = "serif"
rcParams["font.size"] = 10

FILE = "true_vs_predictions.npy"


def interpercentile_plot(
    nets,
    dataset_path,
    list_of_feeder_options,
    plot_type=None,
    delta_perc=[75, 50, 25],
    titles=None,
):

    relative_error = False
    normalize = False
    normal_plot = False

    fig, axes = plt.subplots(1, len(nets), sharey=False, figsize=(6,6))
    if len(nets) == 1:
        axes = [axes]
    if plot_type is None:

        axes[0].set_ylabel("Predicted height [m]")
        normal_plot = True

    elif plot_type == "relative_error":

        axes[0].set_ylabel("Relative error")
        relative_error = True

    elif plot_type == "normalized":

        axes[0].set_ylabel("Normalized prediction [a.u.]")
        normalize = True



    if titles is not None:
        ax_titles = titles 
    else:
        ax_titles = [net.path for net in nets]

    for net, ax, feeder_options , ax_title in zip(nets, axes, list_of_feeder_options, ax_titles):

        feeder_options["shuffle"] = False
        feeder = DataFeeder(dataset_path, **feeder_options)

        model = net.model

        res = net.resolution_on(feeder)

        predictions = model.predict(feeder).squeeze()
        true_vals = np.array([batch[1] for batch in track(feeder)]).reshape((-1))
        ## Set to relative errors
        if relative_error:
            predictions /= true_vals
            predictions -= 1
        elif normalize:
            predictions /= true_vals
            ax.axhline(1, ls=":", color="k")

        # Plots points
        ax.scatter(true_vals, predictions, s=6.0, alpha=0.1, color="k")

        # Number of intervals for percentile estimate
        N = 40

        # Divides the interval of interest (over true values) into N subregions
        # Then takes the values of predictions that fall into that interval
        # And computes the percentile of the sample
        vals = np.linspace(650, 1000, N)
        ups = np.zeros((len(delta_perc), N))
        downs = np.zeros((len(delta_perc), N))
        colormap = cm.get_cmap("plasma")
        print(f"correlation {pearson(true_vals, predictions)}")
        # TEXT FOR PEARSON COEFFICIENT
        ax.text(0.2, 0.9,
                f'Pearson c. = {pearson(true_vals, predictions)}',
                fontsize=12.5,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        # Interval percentile estimation
        for i, delta_perc in enumerate(delta_perc):
            for segment in track(
                range(N - 1), description=f"{delta_perc}-interpercentile:"
            ):
                mask = (true_vals >= vals[segment]) & (true_vals < vals[segment + 1])
                down, up = np.percentile(
                    predictions[mask], [50 - delta_perc / 2, 50 + delta_perc / 2]
                )
                m = np.mean(predictions[mask])
                ups[i, segment] = up
                downs[i, segment] = down

        # Fill each band
        for i in range(len(delta_perc) - 1):
            ax.fill_between(
                vals[:-1],
                downs[i, :-1],
                downs[i + 1, :-1],
                color=colormap(1 - delta_perc[i] / 100.0),
                alpha=0.5,
                label=f"{delta_perc[i]}",
            )
            ax.fill_between(
                vals[:-1],
                ups[i, :-1],
                ups[i + 1, :-1],
                color=colormap(1 - delta_perc[i] / 100.0),
                alpha=0.5,
            )
        # The central one connects ups and downs
        # and must be done by hand
        ax.fill_between(
            vals[:-1],
            ups[-1, :-1],
            downs[-1, :-1],
            label=f"{delta_perc[-1]}",
            color=colormap(1 - delta_perc[-1] / 100.0),
            alpha=0.5,
        )
        ax.legend()

        if normal_plot:
            x_ = np.linspace(650, 1000, 4)
            ax.plot(x_, x_, ls=":", color ='k')

        ax.set_xlabel("True height [m]")

        ax.set_xlim(640, 1010)
        # ax.set_ylim(-0.25, 0.25)
        ax.set_title(f"{ax_title} (res. = {res :.1f} m)")

def pearson(true, predicted):
    """Calculates correlation (Perason Coefficient) between predictions and true values."""
    rho = np.corrcoef(true, predicted)
    return rho[0,1]

if __name__ == "__main__":
    # Call the nets
    lstmenc_vanilla = LstmEncoder(path="trained/lstmenc_train_sub")
    lstmenc_aug = LstmEncoder(path="trained/lstmenc_aug")

    BATCH_SIZE = 128

    # Feeder options for both
    feeder_options = {
        "shuffle": True,
        "batch_size": BATCH_SIZE,
        "input_fields": ["toa", "time_series"],
        "target_field": "outcome",
    }

    # call same vanilla test dataset for both
    train_feeder = DataFeeder("data_by_entry/train", **feeder_options)
