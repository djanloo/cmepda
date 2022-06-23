"""Script to split the dataset

Has to be executed once
"""
from pd4ml import Airshower
from matplotlib import pyplot as plt 
import os
from os.path import join
import utils
THINNING = 1 # Thinning factor: takes one sample in every fixed number

# Load the test dataset
x_test, y_test = Airshower.load_data("test")
toa_test =  x_test['features'][0][::THINNING,:,-1].reshape((-1,9,9,1))
y_test = y_test[::THINNING]

# Clean useless big file
del x_test

test_directory = utils.split_dataset((toa_test, y_test),"test", 100, parent="splitted_dataset")

# Remove saved files from heap
del toa_test, y_test

# Load the training dataset
x_train,y_train = Airshower.load_data("train")
toa_train = x_train['features'][0][::THINNING,:,-1].reshape((-1,9,9,1))
y_train = y_train[::THINNING]
del x_train

train_directory = utils.split_dataset((toa_train, y_train),"train", 100, parent="splitted_dataset")
