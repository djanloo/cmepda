# Make possible importing modules from parent directory
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import matplotlib.pyplot as plt
from cloudatlas import utils
from cloudatlas.utils import DataFeeder
from rich.progress import track
from scipy.stats import gaussian_kde

from matplotlib import rcParams
rcParams['font.family'] = 'serif'


model = utils.ask_load("trained/accuracy_lstm_enc_1120epochs")
if model is None:
    exit("Dumb")

test_feeder = DataFeeder("splitted_dataset/test_splitted_data")

predictions = np.array([])
true_vals = np.array([])

for block in track(test_feeder.feed(), total=test_feeder.n_of_parts):
    predictions = np.concatenate((predictions, model.predict([block["toa"], block["time_series"].reshape((-1, 80, 81))], verbose=0).squeeze() ))
    true_vals = np.concatenate((true_vals, block['outcome']))

data = np.stack((predictions, true_vals), axis=0)
kern = gaussian_kde(data)

x = np.linspace(650, 850, 100)
y = x.copy()

X,Y = np.meshgrid(x,y)
u = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kern(u).T, X.shape)
plt.contourf(X,Y,Z)
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
Z_true = np.tile(Z_true, len(x)).reshape( (-1,) + Z_true.shape)
plt.contourf(X, Y, Z/Z_true )
plt.title("Conditional density")
plt.xlabel("true")
plt.ylabel("estimated")
plt.show()