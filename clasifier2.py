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

#Prepare data
Xmedia = np.concatenate((data_train_media_X,data_test_media_X), axis=0)
Ymedia = np.concatenate((data_train_media_Y,data_test_media_Y), axis=0)
Xzeros = np.concatenate((data_train_zeros_X,data_test_zeros_X), axis=0)
Yzeros = np.concatenate((data_train_zeros_Y,data_test_zeros_Y), axis=0)

Xmedia = np.array(Xmedia).astype(np.float)
Ymedia = np.array(Ymedia).astype(np.int)
Xzeros = np.array(Xzeros).astype(np.float)
Yzeros = np.array(Yzeros).astype(np.int)
#end prepare data

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
Xmedia_pca = pca.fit_transform(Xmedia)
pca = PCA(n_components=3)
Xzeros_pca = pca.fit_transform(Xzeros)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=3)
lda.fit(Xmedia,Ymedia)
Xmedia_lda = lda.transform(Xmedia)
lda = LDA(n_components=3)
lda.fit(Xzeros,Yzeros)
Xzeros_lda = lda.transform(Xzeros)


clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xmedia_pca, Ymedia, cv=5)
results.append(scores.mean())
methods.append('LinearSVC media PCA')
print("LinearSVC Accuracy PCA (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xzeros_pca, Yzeros, cv=5)
results.append(scores.mean())
methods.append('LinearSVC zeros PCA')
print("LinearSVC Accuracy PCA (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xmedia_lda, Ymedia, cv=5)
results.append(scores.mean())
methods.append('LinearSVC media LDA')
print("LinearSVC Accuracy LDA (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xzeros_lda, Yzeros, cv=5)
results.append(scores.mean())
methods.append('LinearSVC zeros LDA')
print("LinearSVC Accuracy LDA (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xmedia_pca, Ymedia, cv=5)
results.append(scores.mean())
methods.append('SVC media PCA')
print("SVC Accuracy PCA (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xzeros_pca, Yzeros, cv=5)
results.append(scores.mean())
methods.append('SVC zeros PCA')
print("SVC Accuracy PCA (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xmedia_lda, Ymedia, cv=5)
results.append(scores.mean())
methods.append('SVC media LDA')
print("SVC Accuracy LDA (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xzeros_lda, Yzeros, cv=5)
results.append(scores.mean())
methods.append('SVC zeros LDA')
print("SVC Accuracy LDA (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
scores = cross_val_score(clf, Xmedia_pca, Ymedia, cv=5)
results.append(scores.mean())
methods.append('Naive media PCA')
print("Naive Accuracy PCA (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xzeros_pca, Yzeros, cv=5)
results.append(scores.mean())
methods.append('Naive zeros PCA')
print("Naive Accuracy PCA (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xmedia_lda, Ymedia, cv=5)
results.append(scores.mean())
methods.append('Naive media LDA')
print("Naive Accuracy LDA (media): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xzeros_lda, Yzeros, cv=5)
results.append(scores.mean())
methods.append('Naive zeros LDA')
print("Naive Accuracy LDA (zeros): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

indices = np.where(results == np.max(results))[0]
print('\n')
print("Best:  "+methods[indices[0]]+" Accuracy: %0.2f" % results[indices[0]])
