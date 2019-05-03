# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import pandas as pd

import sys
# data normalization scripts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.regularizers import l1

def trainNN_Discrete(x_train, y_train, x_test, y_test, hyperparams):
    # keras model
    ny = y_train.shape[1]
    
    model = keras.Sequential()
    for i in range(0,hyperparams['n_layers']):
        model.add(Dense(hyperparams['n_hidden_units'], activation='linear', activity_regularizer = l1(hyperparams['lambda_l1'])))
        model.add(Activation('relu'))
    
    model.add(Dense(ny, activation = 'sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(lr = hyperparams['alpha'], beta_1=0.9, beta_2=0.999, epsilon=1e-8)
      
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['mae', 'mse', 'accuracy'])
    
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0: print(epoch)
    
    history = model.fit(x_train, y_train, epochs = hyperparams['epochs'], batch_size = 64, 
                        validation_data = (x_test, y_test),
                        verbose = 0, callbacks = [PrintDot()])
    
    return model, history

def trainNN_Cost(x_train, y_train, x_test, y_test, hyperparams):
    # keras model
    ny = y_train.shape[1]
    
    model = keras.Sequential()
    for i in range(0,hyperparams['n_layers']):
        model.add(Dense(hyperparams['n_hidden_units'], activation='linear', activity_regularizer = l1(hyperparams['lambda_l1'])))
        model.add(Activation('relu'))
    
    model.add(Dense(ny, activation = 'linear'))
    
    optimizer = tf.keras.optimizers.Adam(lr = hyperparams['alpha'], beta_1=0.9, beta_2=0.999, epsilon=1e-8)
      
    #mse for normal
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae', 'mse', 'accuracy'])
    
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0: print(epoch)
    
    history = model.fit(x_train, y_train, epochs = hyperparams['epochs'], batch_size = 64, 
                        validation_data = (x_test, y_test),
                        verbose = 0, callbacks = [PrintDot()])
    
    return model, history