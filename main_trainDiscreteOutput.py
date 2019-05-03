# -*- coding: utf-8 -*-

# Note: This script uses code from the following sources
# https://www.tensorflow.org/tutorials/keras/basic_regression
# https://stackoverflow.com/questions/1347791/unicode-error-unicodeescape-codec-cant-decode-bytes-cannot-open-text-file

# Description: 
# Trains a neural network to match the binary variable output of a parametric mixed-integer quadratic program

import json
import numpy as np
import pandas as pd
import scipy.io as sio
import sys
# data normalization scripts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# my local scripts
import train

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.regularizers import l1

np.random.seed(0)

#load the data from MATLAB files
matFile = r'C:\Users\guthrjd1\Documents\Graduate Studies\JHU AMS Deep Learning for Discrete Optimization\Project\Final Code\vehicleData_Discrete.mat'
mat_contents = sio.loadmat(matFile)
# input: parameters (power demand)
x = mat_contents['x']
# output: discrete variables (72 binary variables)
y = mat_contents['c']
y = y.T
x = x.T
# ny: dimension of the output data
ny = y.shape[1]
# number of data points
m = y.shape[0]

#split up into test / train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

# normalize data
standardScalerX = StandardScaler()
x_scaled = standardScalerX.fit_transform(x)
x_train_scaled = standardScalerX.fit_transform(x_train)
x_test_scaled = standardScalerX.transform(x_test)
# normalize output data
standardScalerY = StandardScaler()
y_scaled = standardScalerY.fit_transform(y)
y_train_scaled = standardScalerY.fit_transform(y_train)
y_test_scaled = standardScalerY.transform(y_test)

nbatch = 0
for alpha in [0.005]:#[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
    for n_hidden_units in [20]:#[1, 2, 4, 8, 16, 32, 64]:
        for lambda_l1 in [0]:
            for n_layers in [3]:
                nbatch = nbatch + 1
                hyperparams = {'alpha': alpha, 'lambda_l1': lambda_l1, 'n_layers': n_layers, 'n_hidden_units': n_hidden_units, 'epochs':1000}
                model, history = train.trainNN_Discrete(x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, hyperparams)
            
                result = {'batch': nbatch}
                for i in hyperparams.keys():
                    result[i] = hyperparams[i]
                for i in history.history.keys():
                    result[i] = history.history[i]    

                sio.savemat('result_Discrete_' + str(nbatch), result)    


