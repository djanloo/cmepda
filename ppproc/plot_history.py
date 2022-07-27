import numpy as np
import matplotlib.pyplot as plt
import argparse
from rich import print
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.size"] = 11

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--what', type=str, required=True)
parser.add_argument('--what2', type=str, required=True)
parser.add_argument('--title', type=str, required=True)


args = parser.parse_args()

history = np.load(f"{args.path}/history.npy", allow_pickle=True).item()
plt.plot(history.history[args.what], label="training set")
plt.plot(history.history[args.what2], label="validation set")
plt.legend()
plt.title(f"{args.title}")
plt.xlabel("Epoch")
plt.show()