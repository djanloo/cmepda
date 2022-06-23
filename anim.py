# %%
from pd4ml import Airshower
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

x, y = Airshower.load_data("test", graph=False)

# %%
signals = x["features"][0]
print("mx search")
maxval = np.max(signals)
print("end search")
print(signals.shape)
print(y.shape)

# %%
fig = plt.figure()
canvas = plt.imshow(
    np.random.uniform(0, 1, size=(9, 9)), vmin=-1, vmax=maxval, cmap="plasma"
)
height = plt.text(0, 8, "")


def animate(i):
    height.set_text(f"Event at {y[i//81]:.1f}")
    image = signals[i // 81, :, i % 81].reshape((-1, 9))
    canvas.set_array(image)
    return (
        canvas,
        height,
    )


anim = FuncAnimation(fig, animate, interval=0, blit=True)
plt.show()

# %%
