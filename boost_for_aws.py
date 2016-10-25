#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:12:40 2016

@author: reza
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import theano
import sklearn.preprocessing as pp
#import sklearn.cross_validation as cv
import sklearn.model_selection as ms

#import math
import sklearn.ensemble as en
#import sklearn.pipeline as pip
import sklearn.metrics as metrics
import time
print("Packages loaded")
# loaing and preparing data
data_known = pd.read_csv("https://github.com/srhumir/kaggle-regression/raw/master/train.csv")
loss = data_known["loss"]
loss = np.array(loss)
#loss = loss.astype(theano.config.floatX)
data_known.drop("loss", axis=1, inplace = True)
data_known.drop("id", axis=1, inplace = True)
# submit data
data_submit = pd.read_csv('https://github.com/srhumir/kaggle-regression/raw/master/test.csv')
submit_id = data_submit['id']
data_submit.drop('id', axis=1, inplace=True)
print("Data loaded")

# dummy variables
data_dummy = pd.get_dummies(pd.concat([data_known,data_submit], axis=0))
known_rows = data_known.shape[0]
data_known_dummy = data_dummy.iloc[:known_rows,:]
data_submit_dummy= data_dummy.iloc[known_rows:,:]
del(data_dummy)
X_train, X_test, loss_train, loss_test = ms.train_test_split(data_known_dummy, loss, test_size = .1, random_state=0)
X_val, X_test, loss_val, loss_test = ms.train_test_split(X_test, loss_test, test_size = .20, random_state=0)
#X_val2,
indices = pd.read_csv('https://github.com/srhumir/kaggle-regression/raw/master/indices_noid.csv')
indices.drop(indices.columns[0], inplace= True, axis=1)
print("indices ready")
# choosing features
choose = 250
impvars = indices[:choose]
impvars = np.transpose(impvars)
train_data = X_train.iloc[:, impvars]
std = pp.StandardScaler()
#train_all = 
#train_all = std.fit_transform
train_data = std.fit_transform(train_data)
test_data = X_test.iloc[:,impvars]
test_data = std.transform(test_data)
print("starting nn")
# Neural network
import theano
theano.config.openmp_elemwise_minsize=250
OMP_NUM_THREADS=60
theano.config.openmp = True
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
#import keras
#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
def nn_model():#n1,n2=None,n3=None):
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=100, init='normal', 
                 activation='relu'))
#	model.add(Dense(20, init='normal', 
#                 activation='relu'))
#	model.add(Dense(20, init='normal', 
#                 activation='relu'))
	model.add(Dense(1, init='normal'))

   # Compile model
	model.compile(loss='mse', optimizer='adam')
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

epo = 200
print("numbes of epoch %i" %epo)
nn = KerasRegressor(build_fn=nn_model, nb_epoch=epo, batch_size=10,
                           verbose=0) 
print("seat back and relax. I am training")
boost_nn = en.BaggingRegressor(base_estimator = nn,
                                   n_estimators = 500,
                                   max_samples=1.0, 
                                   max_features=100, 
                                   bootstrap=True, 
                                   warm_start=True, 
                                   n_jobs=-1, 
                                   random_state=0, 
                                   verbose=1
                                   )
t0 = time.time()
boost_nn.fit(train_data, loss_train)
t_boost = (time.time()-t0)/60
print("My error on test")
print(metrics.mean_absolute_error(loss_test, boost_nn.predict(test_data)))
print("My error on train")
print(metrics.mean_absolute_error(loss_train, boost_nn.predict(train_data)))
print("it took %f minutes" %t_boost)
# submit
train_submit = data_submit_dummy.iloc[:,impvars]
train_submit = std.transform(train_submit)
loss_submit = boost_nn.predict(train_submit)
to_submit= pd.DataFrame({'id': submit_id, 'loss' : loss_submit})
to_submit.to_csv('submitboost3.csv', index=False)
