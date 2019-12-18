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
from sklearn.metrics import make_scorer

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
    print(b)
    acc = []
    for i in range(5):
        acc.append(b[i,i]/np.sum(b[i, :]))
    print(acc)
    print(np.mean(acc))

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
print('shape of Feature, Label after oversampling')
print(F_re.shape, L_re.shape)
# SMOTE
F_re_smote, L_re_smote = SMOTE().fit_sample(F_train, L_train)
print('shape of Feature, Label after SMOTE')
print(F_re_smote.shape, L_re_smote.shape)

# svm
clf1 = svm.SVC(C=1, gamma=1, probability=True)
clf1.fit(F_re, L_re)
L_pre10 = clf1.predict(F_test0)
L_pre11 = clf1.predict(F_test1)
L_pre12 = clf1.predict(F_test2)
L_pre13 = clf1.predict(F_test3)
Prob10 = clf1.predict_proba(F_test0)
Prob11 = clf1.predict_proba(F_test1)
Prob12 = clf1.predict_proba(F_test2)
Prob13 = clf1.predict_proba(F_test3)
L_pre1 = clf1.predict(F_re)
Prob1 = clf1.predict_proba(F_re)
print('---------------------------')
print('SVM-re')
print('training')
print(diagnosis(L_re, L_pre1))
print('test0') 
print(diagnosis(L_test0, L_pre10))
print('test1')
print(diagnosis(L_test1, L_pre11)) 
print('test2')      
print(diagnosis(L_test2, L_pre12)) 
print('test3')
print(diagnosis(L_test3, L_pre13))

clf2 = svm.SVC(C=10, gamma=0.01, probability=True)
clf2.fit(F_re_smote, L_re_smote)
L_pre20 = clf2.predict(F_test0)
L_pre21 = clf2.predict(F_test1)
L_pre22 = clf2.predict(F_test2)
L_pre23 = clf2.predict(F_test3)
Prob20 = clf2.predict_proba(F_test0)
Prob21 = clf2.predict_proba(F_test1)
Prob22 = clf2.predict_proba(F_test2)
Prob23 = clf2.predict_proba(F_test3)
L_pre2 = clf2.predict(F_re_smote)
Prob2 = clf2.predict_proba(F_re_smote)
print('---------------------------')
print('SVM-smote')
print('training')
print(diagnosis(L_re_smote, L_pre2))
print('test0')
print(diagnosis(L_test0, L_pre20))
print('test1')
print(diagnosis(L_test1, L_pre21)) 
print('test2')
print(diagnosis(L_test2, L_pre22)) 
print('test3')
print(diagnosis(L_test3, L_pre23))

# RF
clf3 = RandomForestClassifier(n_estimators=50, max_depth=50, max_leaf_nodes=50, random_state=0)
clf3.fit(F_re, L_re)
L_pre30 = clf3.predict(F_test0)
L_pre31 = clf3.predict(F_test1)
L_pre32 = clf3.predict(F_test2)
L_pre33 = clf3.predict(F_test3)
Prob30 = clf3.predict_proba(F_test0)
Prob31 = clf3.predict_proba(F_test1)
Prob32 = clf3.predict_proba(F_test2)
Prob33 = clf3.predict_proba(F_test3)
L_pre3 = clf3.predict(F_re)
Prob3 = clf3.predict_proba(F_re)
print('---------------------------')
print('RF-re')
print('training')
print(diagnosis(L_re, L_pre3))
print('test0')
print(diagnosis(L_test0, L_pre30)) 
print('test1')
print(diagnosis(L_test1, L_pre31)) 
print('test2')
print(diagnosis(L_test2, L_pre32))  
print('test3')
print(diagnosis(L_test3, L_pre33))

clf4 = RandomForestClassifier(n_estimators=500, max_depth=50, max_leaf_nodes=50, random_state=0)
clf4.fit(F_re_smote, L_re_smote)
L_pre40 = clf4.predict(F_test0)
L_pre41 = clf4.predict(F_test1)
L_pre42 = clf4.predict(F_test2)
L_pre43 = clf4.predict(F_test3)
Prob40 = clf4.predict_proba(F_test0)
Prob41 = clf4.predict_proba(F_test1)
Prob42 = clf4.predict_proba(F_test2)
Prob43 = clf4.predict_proba(F_test3)
L_pre4 = clf4.predict(F_re_smote)
Prob4 = clf4.predict_proba(F_re_smote)
print('---------------------------')
print('RF-smote')
print('training')
print(diagnosis(L_re_smote, L_pre4))
print('test0')
print(diagnosis(L_test0, L_pre40))
print('test1')
print(diagnosis(L_test1, L_pre41)) 
print('test2')
print(diagnosis(L_test2, L_pre42)) 
print('test3')
print(diagnosis(L_test3, L_pre43))

# KNN
clf5 = KNeighborsClassifier(n_neighbors=2)
clf5.fit(F_re, L_re)
L_pre50 = clf5.predict(F_test0)
L_pre51 = clf5.predict(F_test1)
L_pre52 = clf5.predict(F_test2)
L_pre53 = clf5.predict(F_test3)
Prob50 = clf5.predict_proba(F_test0)
Prob51 = clf5.predict_proba(F_test1)
Prob52 = clf5.predict_proba(F_test2)
Prob53 = clf5.predict_proba(F_test3)
L_pre5 = clf5.predict(F_re)
Prob5 = clf5.predict_proba(F_re)
print('---------------------------')
print('KNN-re')
print('training')
print(diagnosis(L_re, L_pre5))
print('test0')
print(diagnosis(L_test0, L_pre50)) 
print('test1')
print(diagnosis(L_test1, L_pre51)) 
print('test2')
print(diagnosis(L_test2, L_pre52)) 
print('test3')
print(diagnosis(L_test3, L_pre53))

clf6 = KNeighborsClassifier(n_neighbors=2)
clf6.fit(F_re_smote, L_re_smote)
L_pre60 = clf6.predict(F_test0)
L_pre61 = clf6.predict(F_test1)
L_pre62 = clf6.predict(F_test2)
L_pre63 = clf6.predict(F_test3)
Prob60 = clf6.predict_proba(F_test0)
Prob61 = clf6.predict_proba(F_test1)
Prob62 = clf6.predict_proba(F_test2)
Prob63 = clf6.predict_proba(F_test3)
L_pre6 = clf6.predict(F_re_smote)
Prob6 = clf6.predict_proba(F_re_smote)
print('---------------------------')
print('KNN-smote')
print('training')
print(diagnosis(L_re_smote, L_pre6))
print('test0')
print(diagnosis(L_test0, L_pre60))
print('test1')
print(diagnosis(L_test1, L_pre61)) 
print('test2')
print(diagnosis(L_test2, L_pre62)) 
print('test3')
print(diagnosis(L_test3, L_pre63))



