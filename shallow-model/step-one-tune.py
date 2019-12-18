import numpy as np
import csv
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

def range1():
    for n in 10, 50, 100, 250, 500:
        yield n


def range2():
    for n in 10, 50, 100:
        yield n


def range3():
    for n in 0.01, 0.1, 1, 10:
        yield n


def range4():
    for n in 2, 3, 4, 5, 6, 7, 8, 9, 10:
        yield n

def range5():
    for n in 10, 20, 30, 40, 50:
        yield n

'''
def average_auprc(prob, oringin_lb):
    lb = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    lb.fit(oringin_lb)
    binary_lb = lb.transform(oringin_lb)
    a0 = metrics.average_precision_score(binary_lb[:, 0], prob[:, 0])
    a1 = metrics.average_precision_score(binary_lb[:, 1], prob[:, 1])
    a2 = metrics.average_precision_score(binary_lb[:, 2], prob[:, 2])
    a3 = metrics.average_precision_score(binary_lb[:, 3], prob[:, 3])
    a4 = metrics.average_precision_score(binary_lb[:, 4], prob[:, 4])
    print((a0+a1+a2+a3+a4)/5)
    #print(a0, a1, a2, a3, a4)
    return
'''

def diagnosis(l_true, l_pre):
    lb1 = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    lb2 = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    lb1.fit(l_true)
    lb2.fit(l_true)
    binary_true = lb1.transform(l_true)
    binary_pre = lb1.transform(l_pre)
    b = []
    for n in range(5):                        #colunm num
        true = binary_true[:, n]
        for m in range(5):
            pre = binary_pre[:, m]
            a = 0
            for k in range(true.shape[0]):      #row num
                if true[k] == 1 and pre[k] == 1:
                    a += 1
            b.append(a)
    b = np.array(b)
    b = b.reshape([5, 5])
    #print(b)
    acc = []
    for i in range(5):
        acc.append(b[i,i]/np.sum(b[i, :]))
    return(np.mean(acc))


A1 = np.loadtxt('first_layer_sps.txt')
A2 = np.loadtxt('second_layer_sps.txt')
A = np.concatenate((A1, A2), axis=1)            
B1 = np.loadtxt('first_layer_smile.txt')
B2 = np.loadtxt('second_layer_smile.txt')
B = np.concatenate((B1, B2), axis=1)     
F = np.concatenate((A, B), axis=1)
F = preprocessing.scale(F)
L = np.loadtxt('Label_01234')
print('shape of Feature, Label')
print(F.shape, L.shape)

a = np.loadtxt('train_dev_index', dtype='int')
b = np.loadtxt('test0_index', dtype='int')
c = np.loadtxt('test1_index', dtype='int')
d = np.loadtxt('test2_index', dtype='int')
e = np.loadtxt('test3_index', dtype='int')
print('num of train, test0, test1, test2, test3')
print(a.shape, b.shape, c.shape, d.shape, e.shape)

F_train = F[a, :]
L_train = L[a]
F_test0 = F[b, :]
L_test0 = L[b]
F_test1 = F[c, :]
L_test1 = L[c]
F_test2 = F[d, :]
L_test2 = L[d]
F_test3 = F[e, :]
L_test3 = L[e]

# oversampling
ros = RandomOverSampler(random_state=0)
F_re, L_re = ros.fit_sample(F_train, L_train)
# SMOTE
F_re_smote, L_re_smote = SMOTE().fit_sample(F_train, L_train)


#self-define-scorer
self_scorer = make_scorer(diagnosis, greater_is_better = True)


#hyperparameter_tuning
print('---------------------------')
print('hyperparameter tuning results for SVM oversampling, SMOTE')
print('max score, corresponding hyperparameter: C, gamma')
score1 = []
i1 = []
for i in range3():
    for j in range3():
        clf1 = svm.SVC(C=i, gamma=j, probability=True)
        scores = cross_val_score(estimator = clf1, X=F_re, y=L_re, cv=5, scoring=self_scorer)
        scores = scores.mean()
        score1.append(scores)
        i1.append((i, j))
print(max(score1), i1[score1.index(max(score1))])

score2 = []
i2 = []
for i in range3():
    for j in range3():
        clf2 = svm.SVC(C=i, gamma=j, probability=True)
        scores = cross_val_score(clf2, F_re_smote, L_re_smote, cv=5, scoring=self_scorer)
        scores = scores.mean()
        score2.append(scores)
        i2.append((i, j))
print(max(score2), i2[score2.index(max(score2))])


print('---------------------------')
print('hyperparameter tuning results for RF oversampling, SMOTE')
print('max score, corresponding hyperparameter: n_estimators, max_depth')
score3 = []
i3 = []
for i in range1():
    for j in range2():
        for k in range5():
            clf3 = RandomForestClassifier(n_estimators=i, max_depth=j, max_leaf_nodes=k, random_state=0)
            scores = cross_val_score(clf3, F_re, L_re, cv=5, scoring=self_scorer)
            scores = scores.mean()
            score3.append(scores)
            i3.append((i,j,k))
print(max(score3), i3[score3.index(max(score3))])

score4 = []
i4 = []
for i in range1():
    for j in range2():
        for k in range5():
            clf4 = RandomForestClassifier(n_estimators=i, max_depth=j, max_leaf_nodes=k, random_state=0)
            scores = cross_val_score(clf4, F_re_smote, L_re_smote, cv=5, scoring=self_scorer)
            scores = scores.mean()
            score4.append(scores)
            i4.append((i,j,k))
print(max(score4), i4[score4.index(max(score4))])


print('---------------------------')
print('hyperparameter tuning results for KNN oversampling, SMOTE')
print('max score, corresponding hyperparameter: n_neighbors')
score5 = []
i5 = []
for i in range4():
    clf5 = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(clf5, F_re, L_re, cv=5, scoring=self_scorer)
    scores = scores.mean()
    score5.append(scores)
    i5.append(i)
print(max(score5), i5[score5.index(max(score5))])

score6 = []
i6 = []
for i in range4():
    clf6 = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(clf6, F_re_smote, L_re_smote, cv=5, scoring=self_scorer)
    scores = scores.mean()
    score6.append(scores)
    i6.append(i)
print(max(score6), i6[score6.index(max(score6))])

