"""Utility module"""
import os
import warnings

# Turn off keras warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from keras.models import load_model
from rich import print
import telegram_send


class RemoteMonitor:
    """Class for remote logging using telegram."""

    def __init__(self):
        try:
            import telegram_send

            print(f"Remote monitor [green]available[/green]")
        except ImportError:
            print(f"Remote monitor [red]unavailable[/red]")

        # Check if telegram_send is configured
        # ????

    def send(self, message):
        """Sends a message."""
        # Check wether it's a message or more than one
        if hasattr(message, "__iter__"):
            msg = message
        else:
            msg = [message]
        try:
            telegram_send.send(messages=msg)
        except Exception as e:
            warnings.warn("Network failed while remote monitoring.")
            print(e)


def ask_load(path):
    """Conditionally loads a saved Model.

    Arguments
    ---------
        path : str
            the path of the Model
    """
    if os.path.exists(path):
        print(
            f"Existing model found at [green]{path}[/green]. Do you want to load it? [blue](y/n)"
        )
        ans = input()
        if ans == "y":
            return load_model(path)
        elif ans == "n":
            return None
        else:
            return ask_load(path)
    else:
        return None


def animate_time_series(array):
    """Animate the time series for detectors."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    vmax, vmin = np.max(array), np.min(array)
    fig = plt.figure()
    canvas = plt.imshow(
        np.random.uniform(0, 1, size=(9, 9)), vmin=vmin, vmax=vmax, cmap="plasma"
    )

    def animate(i):
        image = array[i]
        canvas.set_array(image)
        return (canvas,)

    return FuncAnimation(fig, animate, frames=len(array), interval=0, blit=True)
