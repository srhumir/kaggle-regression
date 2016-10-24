# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:24:54 2016

@author: Hanireza
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
#__location__ = os.path.realpath(
#    os.path.join(os.getcwd(), os.
#sys.argv[0]
#))
#file = zip.ZipFile('train.csv.zip', "c:/Reza/python/regression/Kaggle")
#file.namelist()
#file.extractall()
#import urllib2
#response = urllib2.urlopen('https://github.com/srhumir/kaggle-regression/raw/master/train.csv')
#html = response.read()
data_known = pd.read_csv("https://github.com/srhumir/kaggle-regression/raw/master/train.csv")
loss = data_known["loss"]
loss = np.array(loss)
#loss = loss.astype(theano.config.floatX)
data_known.drop("loss", axis=1, inplace = True)
data_known.drop("id", axis=1, inplace = True)



# dummy variables
data_known_dummy = pd.get_dummies(data_known)
X_train, X_test, loss_train, loss_test = ms.train_test_split(data_known_dummy, loss, test_size = .3, random_state=0)
X_val, X_test, loss_val, loss_test = ms.train_test_split(X_test, loss_test, test_size = .20, random_state=0)
#X_val2, X_test, loss_val2, loss_test = ms.train_test_split(X_test, loss_test, test_size = .66, random_state=0)
#X_val3, X_test, loss_val3, loss_test = ms.train_test_split(X_test, loss_test, test_size = .5, random_state=0)
#==============================================================================
# #load saved train data
# X_train = pd.read_csv("X_train.csv")
# loss_train = pd.read_csv("loss_train.csv")
# id_train = X_train["id"]
# X_train.drop("Unnamed: 0", axis=1, inplace = True)
# X_train.drop("id", axis=1, inplace = True)
# loss_train.drop("Unnamed: 0", axis=1, inplace = True)
# loss_train = np.ravel(loss_train.values)
# indices = pd.read_csv("indices.csv")
# indices.drop("Unnamed: 0", axis=1, inplace = True)
# indices = np.ravel(indices.values)
# 
# 
#==============================================================================

## feature selection
feat_lables = X_train.columns

rf = en.RandomForestRegressor(n_estimators = 100,
                              random_state = 1,
                              n_jobs= -1)
t0_rf = time.time()
rf.fit(X_train, loss_train)
t_rf = time.time() - t0_rf

imps = rf.feature_importances_
indices = np.argsort(imps)[::-1]
cumsum = np.cumsum(imps[indices])
for f in range(50):
    print("%2d) %-*s %f %g" % (f + 1, 30,feat_lables[indices[f]],
                                        imps[indices[f]],
                                        cumsum[f]))

#plt.scatter(X_train["cont7"], loss_train)
#plt.hist(loss, bins=1000)


#==============================================================================
# a = (1-.999)/20
# thress = [.99+a*i for i in range(20)]
# #for i in range(20):
# #    thress.append(thress[-1]+a)
# 
# thress = np.array(range(999,99999))/100000
# thress = np.append(thress, .999)
# number_vars=[]
# train_err = []
# cv_err =[]
# #threshold = .79
# #val_err=[]
# t0_featselec = time.time()
# for threshold in thress:
#     choose = np.where(cumsum >= threshold)[0][0]
#     impvars = indices[:choose]
#     err = cv.cross_val_score(estimator=rf,
#                           X=X_train.iloc[:,impvars],
#                           y=loss_train,
#                           scoring = 'mean_absolute_error',
#                           cv=5,
#                           n_jobs=-1)
# #    rf.fit(np.array(X_train)[:, impvars], loss_train)
#     number_vars.append(choose+1)
#     cv_err.append(err)
#     print(threshold)
#     print('So far %d seconds' %(time.time() - t0_featselec))
# print('Total time: %d seconds' %(time.time()-t0_featselec))
#==============================================================================
#    val_err.append(math.sqrt(metrics.mean_squared_error(loss_val,rf.predict(X_val_std[:, impvars]))))
#chooses = [int(X_train.shape[1]/40)*i for i in range(20)]
def check_imp(chooses, rfmodel, X_train, y_train, indices, X_val, y_val):
    cv_err =[]
    kf = ms.KFold(n_splits=4, random_state=0)
    t0_featselec = time.time()
    for choose in chooses:
        impvars = indices[:choose+1]
        rf.fit(X_train.iloc[:,impvars], y=y_train, )
        err =[]
        for train, test in kf.split(X_val):
            err.append(metrics.mean_absolute_error(y_val[test], rf.predict(X_val.iloc[test,impvars])))
#==============================================================================
#         err = ms.cross_val_score(estimator=rfmodel,
#                                  X=X_train.iloc[:,impvars],
#                                  y=y_train,
#                           scoring = 'neg_mean_absolute_error',
#                           cv=4,
#                           n_jobs=-1)
#==============================================================================
        err=np.transpose(err)
        cv_err.append(err)
        print('With %d variables. Step %d of %d' %((choose+1), np.where(np.array(chooses)==choose)[0]+1,len(chooses)))
        print('So far %d seconds' %(time.time() - t0_featselec))
    print('Total time: %d seconds' %(time.time()-t0_featselec))
    return(np.array(cv_err))

#cv_err1 = cv_err    
#chooses1 = chooses


def importance_plot(chooses, cv_err):
    cv_err = np.array(cv_err)
    fig=plt.figure(figsize=(10,10))
    plt.subplot()
    plt.plot(chooses, cv_err.mean(axis=1),
             color='blue', marker='o',
             label='Cross validation error')
    plt.fill_between(chooses,
                     cv_err.mean(axis=1) - cv_err.std(axis=1),
                     cv_err.mean(axis=1) + cv_err.std(axis=1),
                     alpha=0.3, color='green')
#   plt.plot(number_vars, val_err,
#            color='green', marker='s',
#            label='Validation error')
    plt.grid()
    plt.xlabel('Number of vraibles in the model')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance of the RF model by number of features')
    plt.legend(loc=8)
    plt.show()

no_points = 20
steps = int(X_train.shape[1]/no_points) 
chooses1 = [int(X_train.shape[1]/40)*i for i in range(10)]
#chooses1 = [steps*i for i in range(no_points+1)]
cv_err1 = check_imp(chooses1, rf, X_train, loss_train, indices, X_val, loss_val)
importance_plot(np.array(chooses1)+1, cv_err1)

# smaller
steps2 = 3
chooses2 = [5*steps + i * steps2 for i in range(20)]
cv_err2 = check_imp(chooses2, rf, X_train, loss_train, indices)
importance_plot(chooses2, cv_err2)
importance_plot(np.append(chooses1,chooses2), np.append(cv_err1, cv_err2, axis=0))

########### The number of important values is 140 considering trade-off between std and mean error
pd.DataFrame(indices).to_csv("indices_noid.csv")
choose = 140
impvars = indices[:choose]
train_data = X_train.iloc[:, impvars]
std = pp.StandardScaler()
train_data = std.fit_transform(train_data)
#trying different models
## RF
rf = en.RandomForestRegressor(n_estimators = 1000,
                              random_state = 1,
                              n_jobs= -1)
rf_err1000 = ms.cross_val_score(estimator=rf,
                         X=train_data,
                         y=loss_train,
                         scoring = 'neg_mean_absolute_error',
                         cv=5,
                         n_jobs=-1)
#print("Error = %i +/- %i" %(-rf_err200.mean(), int(2*rf_err200.std())))
#rf_err = pd.DataFrame({'100':rf_err100,'200':rf_err200,'300':rf_err300,'500':rf_err500, '1000':rf_err1000})
#rf_err.to_csv("rf_error.csv")
#print("R-squared = %f" 
#      %metrics.r2_score(loss_train, rf.predict(train_data)))

# Neural network
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
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
#from sklearn.model_selection import cross_val_score
kfold = ms.KFold(n_splits=10, random_state=seed)

#results = cross_val_score(estimator, X, Y, cv=kfold)

t0_nn = time.time()
nn = KerasRegressor(build_fn=nn_model, nb_epoch=200, batch_size=5,
                           verbose=0) 
#nn.fit(train_data, loss_train,
#       validation_split=0.1,
#       show_accuracy=True)
#nn_time = time.time()-t0_nn

t0_nn= time.time()
nn_err_280 = ms.cross_val_score(nn,
                            X=train_data,
                            y=loss_train,
                            scoring = 'neg_mean_absolute_error',
                            cv=kfold,
                            n_jobs=-1)
nn_time = time.time()-t0_nn
#nn_err = pd.DataFrame({'20':nn_err1_20, '100':nn_err_100, '200':nn_err_200, '20-20':nn_err_20_20, '20-50':nn_err_20_50})
#nn_err.to_csv("nn_error.csv")

#kernel ridge regression
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import time
kr = KernelRidge(kernel='rbf', gamma=0.1)
gs_kr = GridSearchCV(kr,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)},
                              scoring = 'neg_mean_absolute_error',
                              cv=5,
                              n_jobs=-1)
t0_gs_kr = time.time()
gs_kr.fit(train_data, loss_train)
gs_kr_time = time.time()- t0_gs_kr



