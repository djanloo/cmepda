from lstm_encoder import train_and_resolution
import numpy as np
from rich import print
paths = [f"trained/accuracy_lstm_enc_{i}" for i in range(5)]
with open("accuracy", "w") as outfile:
    for path in paths:
        print(f"Training [green]{path}")
        res = train_and_resolution(path)
        outfile.write(f"{res}\n")
