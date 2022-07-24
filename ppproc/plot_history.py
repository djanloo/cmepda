import numpy as np
import matplotlib.pyplot as plt
import argparse
from rich import print
from matplotlib import rcParams

rcParams["font.family"] = "serif"

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--what', type=str, required=True)

args = parser.parse_args()

history = np.load(f"{args.path}/history.npy", allow_pickle=True).item()
plt.plot(history.history[args.what], label=args.what)

plt.title(f"Model at {args.path}")
plt.xlabel("Epoch")
plt.legend()
plt.show()