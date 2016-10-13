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
import sklearn.cross_validation as cv
import math
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


# dummy variables
data_known_dummy = pd.get_dummies(data_known)
X_train, X_test, loss_train, loss_test = cv.train_test_split(data_known_dummy, loss, test_size = .98, random_state=0)

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
chooses = [int(X_train.shape[1]/40)*i for i in range(42)]
def check_imp(chooses, rfmodel, X_train, y_train, indices):
    train_err = []
    cv_err =[]
    t0_featselec = time.time()
    for choose in chooses:
        impvars = indices[:choose+1]
        err = cv.cross_val_score(estimator=rf,
                          X=X_train.iloc[:,impvars],
                          y=loss_train,
                          scoring = 'mean_absolute_error',
                          cv=5,
                          n_jobs=-1)
    cv_err.append(err)
    print(choose)
    print('So far %d seconds' %(time.time() - t0_featselec))
print('Total time: %d seconds' %(time.time()-t0_featselec))

cv_err = np.array(cv_err)
fig=plt.figure(figsize=(10,10))
plt.subplot()
plt.plot(chooses, -cv_err.mean(axis=1),
         color='blue', marker='o',
         label='Cross validation error')
plt.fill_between(chooses,
                 -cv_err.mean(axis=1) - cv_err.std(axis=1),
                 -cv_err.mean(axis=1) + cv_err.std(axis=1),
                 alpha=0.15, color='green')
#plt.plot(number_vars, val_err,
#         color='green', marker='s',
#         label='Validation error')
plt.grid()
plt.xlabel('Number of vraibles in the model')
plt.ylabel('Root mean square erro')
plt.title('Performance of the RF model by number of features')
plt.legend(loc=7)
plt.show()

# choosing variables
#choose the best model
thresh = thress[np.where(val_err == np.min(val_err))[0][-1]]
choose = np.where(cumsum >= thresh)[0][0]
impvars = indices[:choose]
# 12 features where selected
# trying models with selected features
## random forest
rf.fit(X_train_std[:, impvars], target_train)
math.sqrt(metrics.mean_squared_error(target_val, rf.predict(X_val_std[:, impvars])))
metrics.r2_score(target_val, rf.predict(X_val_std[:, impvars]))