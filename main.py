import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.svm import SVC
style.use("ggplot")
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import friedmanchisquare

import receive_data as rdata

results = list()
methods = list()
LinearSVCresults = list()
RBFresults = list()
GaussianNBresults = list()
RandomForestresults = list()
data_train_media_X, data_train_media_Y = rdata.load_data_converted('db_text/train_baches_media_fixtures.txt')
data_test_media_X, data_test_media_Y = rdata.load_data_converted('db_text/test_baches_media_fixtures.txt')

data_train_zeros_X, data_train_zeros_Y = rdata.load_data_converted('db_text/train_baches_zeros_fixtures.txt')
data_test_zeros_X, data_test_zeros_Y = rdata.load_data_converted('db_text/test_baches_zeros_fixtures.txt')


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
LinearSVCresults.append(scores.mean())
methods.append('LinearSVC media')
print("LinearSVC Accuracy (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xzeros, Yzeros, cv=5)
results.append(scores.mean())
LinearSVCresults.append(scores.mean())
methods.append('LinearSVC zeros')
print("LinearSVC Accuracy (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xmedia, Ymedia, cv=5)
results.append(scores.mean())
RBFresults.append(scores.mean())
methods.append('SVC media')
print("SVC Accuracy (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xzeros, Yzeros, cv=5)
results.append(scores.mean())
RBFresults.append(scores.mean())
methods.append('SVC zeros')
print("SVC Accuracy (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xmedia, Ymedia, cv=5)
results.append(scores.mean())
GaussianNBresults.append(scores.mean())
methods.append('Naive media')
print("Naive Accuracy (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xzeros, Yzeros, cv=5)
results.append(scores.mean())
GaussianNBresults.append(scores.mean())
methods.append('Naive zeros')
print("Naive Accuracy (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


clf = RandomForestClassifier(n_estimators=30)
scores = cross_val_score(clf, Xmedia, Ymedia, cv=5)
results.append(scores.mean())
RandomForestresults.append(scores.mean())
methods.append('Random Forest media')
print("Random Forest Accuracy (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = RandomForestClassifier(n_estimators=30)
scores = cross_val_score(clf, Xzeros, Yzeros, cv=5)
results.append(scores.mean())
RandomForestresults.append(scores.mean())
methods.append('Random Forest zeros')
print("Random Forest Accuracy (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

indices = np.where(results == np.max(results))[0]
print('\n')
print("Best:  "+methods[indices[0]]+" Accuracy: %0.3f" % results[indices[0]])

data_train_media_X, data_train_media_Y = rdata.load_data_converted('db_text/train_baches_media_fixtures.txt')
data_test_media_X, data_test_media_Y = rdata.load_data_converted('db_text/test_baches_media_fixtures.txt')

data_train_zeros_X, data_train_zeros_Y = rdata.load_data_converted('db_text/train_baches_zeros_fixtures.txt')
data_test_zeros_X, data_test_zeros_Y = rdata.load_data_converted('db_text/test_baches_zeros_fixtures.txt')

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
LinearSVCresults.append(scores.mean())
methods.append('LinearSVC media PCA')
print("LinearSVC Accuracy PCA (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xzeros_pca, Yzeros, cv=5)
results.append(scores.mean())
LinearSVCresults.append(scores.mean())
methods.append('LinearSVC zeros PCA')
print("LinearSVC Accuracy PCA (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xmedia_lda, Ymedia, cv=5)
results.append(scores.mean())
LinearSVCresults.append(scores.mean())
methods.append('LinearSVC media LDA')
print("LinearSVC Accuracy LDA (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


clf = SVC(kernel='linear', C=1.0)
scores = cross_val_score(clf, Xzeros_lda, Yzeros, cv=5)
results.append(scores.mean())
LinearSVCresults.append(scores.mean())
methods.append('LinearSVC zeros LDA')
print("LinearSVC Accuracy LDA (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xmedia_pca, Ymedia, cv=5)
results.append(scores.mean())
RBFresults.append(scores.mean())
methods.append('SVC media PCA')
print("SVC Accuracy PCA (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xzeros_pca, Yzeros, cv=5)
results.append(scores.mean())
RBFresults.append(scores.mean())
methods.append('SVC zeros PCA')
print("SVC Accuracy PCA (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xmedia_lda, Ymedia, cv=5)
results.append(scores.mean())
RBFresults.append(scores.mean())
methods.append('SVC media LDA')
print("SVC Accuracy LDA (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
scores = cross_val_score(clf, Xzeros_lda, Yzeros, cv=5)
results.append(scores.mean())
RBFresults.append(scores.mean())
methods.append('SVC zeros LDA')
print("SVC Accuracy LDA (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xmedia_pca, Ymedia, cv=5)
results.append(scores.mean())
GaussianNBresults.append(scores.mean())
methods.append('Naive media PCA')
print("Naive Accuracy PCA (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xzeros_pca, Yzeros, cv=5)
results.append(scores.mean())
GaussianNBresults.append(scores.mean())
methods.append('Naive zeros PCA')
print("Naive Accuracy PCA (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xmedia_lda, Ymedia, cv=5)
results.append(scores.mean())
GaussianNBresults.append(scores.mean())
methods.append('Naive media LDA')
print("Naive Accuracy LDA (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = GaussianNB()
scores = cross_val_score(clf, Xzeros_lda, Yzeros, cv=5)
results.append(scores.mean())
GaussianNBresults.append(scores.mean())
methods.append('Naive zeros LDA')
print("Naive Accuracy LDA (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = RandomForestClassifier(n_estimators=30)
scores = cross_val_score(clf, Xmedia_pca, Ymedia, cv=5)
results.append(scores.mean())
RandomForestresults.append(scores.mean())
methods.append('Random Forest media PCA')
print("Random Forest Accuracy PCA (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = RandomForestClassifier(n_estimators=30)
scores = cross_val_score(clf, Xzeros_pca, Yzeros, cv=5)
results.append(scores.mean())
RandomForestresults.append(scores.mean())
methods.append('Random Forest zeros PCA')
print("Random Forest Accuracy PCA (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = RandomForestClassifier(n_estimators=30)
scores = cross_val_score(clf, Xmedia_lda, Ymedia, cv=5)
results.append(scores.mean())
RandomForestresults.append(scores.mean())
methods.append('Random Forest media LDA')
print("Random Forest Accuracy LDA (media): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

clf = RandomForestClassifier(n_estimators=30)
scores = cross_val_score(clf, Xzeros_lda, Yzeros, cv=5)
results.append(scores.mean())
RandomForestresults.append(scores.mean())
methods.append('Random Forest zeros LDA')
print("Random Forest Accuracy LDA (zeros): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

indices = np.where(results == np.max(results))[0]
print('\n')
print("Best:  "+methods[indices[0]]+" Accuracy: %0.3f" % results[indices[0]])

means = list()
means.append(np.mean(LinearSVCresults))
means.append(np.mean(RBFresults))
means.append(np.mean(GaussianNBresults))
means.append(np.mean(RandomForestresults))
indices = np.where(means == np.max(means))[0]

if indices[0] == 0:
    print("LinearSVC: %0.3f " % (means[0]))
if indices[0] == 1:
    print("RBF: %0.3f " % (means[1]))
if indices[0] == 2:
    print("GaussianNB: %0.3f " % (means[2]))
if indices[0] == 3:
    print("RandomForest: %0.3f " % (means[3]))

measurements1 = LinearSVCresults
measurements2 = RBFresults
measurements3 = GaussianNBresults
measurements4 = RandomForestresults
chi_square, p_value = friedmanchisquare(measurements1, measurements2, measurements3, measurements4)
print(chi_square)
print(p_value)

plt.plot(measurements1)
plt.plot(measurements2)
plt.plot(measurements3)
plt.plot(measurements4)
plt.show()
