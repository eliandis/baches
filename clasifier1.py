import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn.svm import SVC
style.use("ggplot")
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.svm import SVR
import receive_data as rdata

results = list()
methods = list()
data_train_media_X, data_train_media_Y = rdata.load_data_coverted('db_text/train_baches_media_fixtures.txt')
data_test_media_X, data_test_media_Y = rdata.load_data_coverted('db_text/test_baches_media_fixtures.txt')

data_train_zeros_X, data_train_zeros_Y = rdata.load_data_coverted('db_text/train_baches_zeros_fixtures.txt')
data_test_zeros_X, data_test_zeros_Y = rdata.load_data_coverted('db_text/test_baches_zeros_fixtures.txt')


Xmedia = np.concatenate((data_train_media_X,data_test_media_X), axis=0)
Ymedia = np.concatenate((data_train_media_Y,data_test_media_Y), axis=0)
Xzeros = np.concatenate((data_train_zeros_X,data_test_zeros_X), axis=0)
Yzeros = np.concatenate((data_train_zeros_Y,data_test_zeros_Y), axis=0)

Xmedia = np.array(Xmedia).astype(np.float)
Ymedia = np.array(Ymedia).astype(np.int)
Xzeros = np.array(Xzeros).astype(np.float)
Yzeros = np.array(Yzeros).astype(np.int)


clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xmedia, Ymedia, cv=5)
results.append(scores.mean())
methods.append('LinearSVC media')
print("LinearSVC Accuracy (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xzeros, Yzeros, cv=5)
results.append(scores.mean())
methods.append('LinearSVC zeros')
print("LinearSVC Accuracy (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xmedia, Ymedia, cv=5)
results.append(scores.mean())
methods.append('SVC media')
print("SVC Accuracy (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xzeros, Yzeros, cv=5)
results.append(scores.mean())
methods.append('SVC zeros')
print("SVC Accuracy (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
scores = cross_val_score(clf, Xmedia, Ymedia, cv=5)
results.append(scores.mean())
methods.append('Naive media')
print("Naive Accuracy (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xzeros, Yzeros, cv=5)
results.append(scores.mean())
methods.append('Naive zeros')
print("Naive Accuracy (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

indices = np.where(results == np.max(results))[0]
print('\n')
print("Best:  "+methods[indices[0]]+" Accuracy: %0.2f" % results[indices[0]])

import scipy.stats as est
est.friedmanchisquare()
