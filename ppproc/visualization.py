import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from context import utils as ut
from context import DataFeeder

def animate_time_series(array):
    """Animate the time series for detectors."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    vmax, vmin = np.max(array), np.min(array)
    fig = plt.figure()
    canvas = plt.imshow(
        np.random.uniform(0, 1, size=(9, 9)), vmin=vmin, vmax=vmax, cmap="viridis"
    )

    def animate(i):
        image = array[i]
        canvas.set_array(image)
        return (canvas,)

    return FuncAnimation(fig, animate, frames=len(array), interval=1, blit=True)


if __name__ == "__main__":
    np.random.seed(42)
    rcParams["font.family"] = "serif"

    feeder_options = {
        "shuffle": True, # Only for testing
        "batch_size": 11,
        "input_fields": ["toa", "time_series"],
        "target_field": "outcome",
    }

    df = DataFeeder("data_by_entry/train", **feeder_options)
    ts = df[0][0][1][:].reshape((-1, 9,9))
    toa = df[0][0][0]
    print(ts.shape)

    fig, axes = plt.subplots(2,2)
    c=0
    for row in axes:
        for col in row:
            u=col.imshow(toa[c])
            col.axis("off")
            plt.colorbar(u,ax=col)
            c+=1
    fig.suptitle("Times of arrival")
    fig.tight_layout()
    plt.show()
    # plt.figure(2)
    # anim = ut.animate_time_series(ts)
    # plt.axis('off')
    # plt.title("Detector time series")
    # anim.save("anim.gif")
