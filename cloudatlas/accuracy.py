from lstm_encoder import train_and_resolution
import numpy as np
from rich import print
import telegram_send

paths = [f"trained/accuracy_lstm_enc_{i}" for i in range(5)]

with open("accuracy", "w") as outfile:
    for path in paths:
        print(f"Training [green]{path}")
        res = train_and_resolution(path)
        outfile.write(f"{res}\n")
        try:
            telegram_send.send(messages=[f"Training of {path} finished with <u>resolution = {res:.1f}</u>"], parse_mode="HTML")
        except:
            print("Network failed")

a = np.loadtxt("accuracy")
try:
    telegram_send.send(messages=[f"Training finished with <u>resolution = {np.mean(a)} +- {np.std(a)}</u>"], parse_mode="HTML")
except:
    print("Network failed")