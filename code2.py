# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:35:54 2016

@author: sreza
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import theano
import sklearn.preprocessing as pp
import sklearn.cross_validation as cv
#import math
#import sklearn.ensemble as en
#import sklearn.pipeline as pip
import sklearn.metrics as metrics
import time
#import keras

data_known = pd.read_csv("https://github.com/srhumir/kaggle-regression/raw/master/train.csv")
loss = data_known["loss"]
loss = np.array(loss)
#loss = loss.astype(theano.config.floatX)
data_known.drop("loss", axis=1, inplace = True)
data_known.drop("id", axis=1, inplace = True)


# dummy variables
data_known_dummy = pd.get_dummies(data_known)
X_train, X_test, loss_train, loss_test = cv.train_test_split(data_known_dummy, loss, test_size = .2, random_state=0)
#load saved variable importance
indices = pd.read_csv("indices.csv")
indices.drop("Unnamed: 0", axis=1, inplace = True)
indices = np.ravel(indices.values)

feat_lables = X_train.columns

choose = 140
impvars = indices[:choose]
train_data = X_train.iloc[:, impvars]
std = pp.StandardScaler()
train_data = std.fit_transform(train_data)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
def nn_model():#n1,n2=None,n3=None):
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=impvars.size, init='normal', 
                 activation='relu'))
#	model.add(Dense(20, init='normal', 
#                 activation='relu'))
#	model.add(Dense(20, init='normal', 
#                 activation='relu'))
	model.add(Dense(1, init='normal'))

   # Compile model
	model.compile(optimizer='rmsprop',
              loss='mse')
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score

t0_nn = time.time()
nn = KerasRegressor(build_fn=nn_model, nb_epoch=200, batch_size=5,
                           verbose=1) 
nn.fit(train_data, loss_train,
       validation_split=0.1)
nn_time = time.time()-t0_nn