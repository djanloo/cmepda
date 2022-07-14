import numpy as np
import matplotlib.pyplot as plt
import argparse
from rich import print

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

history = np.load(f"{args.path}/history.npy", allow_pickle=True).item()
for key in history.history:
    plt.plot(history.history[key], label=key)

plt.title(f"Model at {args.path}")
plt.xlabel("Epoch")
plt.legend()
plt.show()