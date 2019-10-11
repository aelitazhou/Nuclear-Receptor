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

def myrange():
    for n in 100, 500, 1000, 1500, 2000, 3000, 4000, 5000:
        yield n


def mmyrange():
    for n in 0.01, 0.1, 1, 10:
        yield n


def myrangee():
    for n in 3, 4, 5, 6, 7, 8, 9, 10:
        yield n


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


with open('step-one.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    S = []
    for i in csv_reader:
        S.append(i)
        SS = np.array(S)
with open('uniqueNRs.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    U = []
    for i in csv_reader:
        U.append(i)
        UU = np.array(U)

A1 = np.loadtxt('first_layer_states_fasta.txt')
A2 = np.loadtxt('second_layer_states_fasta.txt')
A = np.concatenate((A1, A2), axis=1)             #feature of 28 pros
B1 = np.loadtxt('first_layer_states_smile.txt')
B2 = np.loadtxt('second_layer_states_smile.txt')
B = np.concatenate((B1, B2), axis=1)             #feature of 3711 smiles
C = UU[1:29, 2]      #name of the 28 pro
D = SS[1:3730, 0]    #name of the 3729 pro
E = SS[1:3730, 2]    #label 3729

Pro = []
index = []
for i in range(3729):
    for j in range(28):
        if D[i] == C[j]:
            Pro.append(A[j])
            index.append(i)
index = np.array(index)
#indexw = np.arange(0, 3729, 1)
#excep = np.delete(indexw, index, axis=0)   #no exceptions  excpe=[]
L = []
e = []
for i in range(3729):
    if E[i] == 'agonist':
        L.append('0')
    elif E[i] == 'partial agonist':
        L.append('1')
    elif E[i] == 'inverse agonist':
        L.append('2')
    elif E[i] == 'antagonist':
        L.append('3')
    elif E[i] == 'partial antagonist':
        L.append('4')
    else:
        e.append(i)
L = np.array(L)
Pro = np.array(Pro)
P = np.delete(Pro, e, axis=0)
print(P.shape, B.shape)
F = np.concatenate((P, B), axis=1)        #(3711, 1024)
F = preprocessing.scale(F)

#training_test splitting
a = np.load('train_index.npy')
b = np.load('test1_index.npy')
c = np.load('test2_index.npy')
d = np.load('test3_index.npy')
F_train = F[a, :]
L_train = L[a]
F_train, F_test0, L_train, L_test0 = train_test_split(F_train, L_train, test_size=0.1)
F_test1 = F[b, :]
L_test1 = L[b]
F_test2 = F[c, :]
L_test2 = L[c]
F_test3 = F[d, :]
L_test3 = L[d]

#oversampling
ros = RandomOverSampler(random_state=0)
F_re, L_re = ros.fit_sample(F_train, L_train)

#SMOTE
F_re_smote, L_re_smote = SMOTE().fit_sample(F_train, L_train)
print(L_test1.shape, L_test2.shape, L_test3.shape, L_re.shape, L_re_smote.shape, L_test0.shape)
'''
#hyperparameter_tuning(before split out the test0 set)
score1 = []
i1 = []
for i in mmyrange():
    for j in mmyrange():
        clf1 = svm.SVC(C=i, gamma=j, probability=True)
        scores = cross_val_score(clf1, F_re, L_re, cv=5, scoring='accuracy')
        scores = scores.mean()
        score1.append(scores)
        i1.append((i, j))
print(max(score1), i1[score1.index(max(score1))])

score2 = []
i2 = []
for i in mmyrange():
    for j in mmyrange():
        clf2 = svm.SVC(C=i, gamma=j, probability=True)
        scores = cross_val_score(clf2, F_re_smote, L_re_smote, cv=5, scoring='accuracy')
        scores = scores.mean()
        score2.append(scores)
        i2.append((i, j))
print(max(score2), i2[score2.index(max(score2))])

score3 = []
i3 = []
for i in myrange():
    clf3 = RandomForestClassifier(n_estimators=i, max_depth=None, random_state=0)
    scores = cross_val_score(clf3, F_re, L_re, cv=5, scoring='accuracy')
    scores = scores.mean()
    score3.append(scores)
    i3.append(i)
print(max(score3), i3[score3.index(max(score3))])

score4 = []
i4 = []
for i in myrange():
    clf4 = RandomForestClassifier(n_estimators=i, max_depth=None, random_state=0)
    scores = cross_val_score(clf4, F_re_smote, L_re_smote, cv=5, scoring='accuracy')
    scores = scores.mean()
    score4.append(scores)
    i4.append(i)
print(max(score4), i4[score4.index(max(score4))])

score5 = []
i5 = []
for i in myrangee():
    clf5 = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(clf5, F_re, L_re, cv=5, scoring='accuracy')
    scores = scores.mean()
    score5.append(scores)
    i5.append(i)
print(max(score5), i5[score5.index(max(score5))])

score6 = []
i6 = []
for i in myrangee():
    clf6 = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(clf6, F_re_smote, L_re_smote, cv=5, scoring='accuracy')
    scores = scores.mean()
    score6.append(scores)
    i6.append(i)
print(max(score6), i6[score6.index(max(score6))])
'''

clf1 = svm.SVC(C=10, gamma=0.01, probability=True)
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
print('SVM: auprc and accuracy rate for the four test sets')
print(average_auprc(Prob10, L_test0),average_auprc(Prob11, L_test1), average_auprc(Prob12, L_test2), average_auprc(Prob13, L_test3))
print(accuracy_score(L_test0, L_pre10),accuracy_score(L_test1, L_pre11), accuracy_score(L_test2, L_pre12), accuracy_score(L_test3, L_pre13))
print('SVM: auprc and accuracy rate for the training set')
print(average_auprc(Prob1, L_re), accuracy_score(L_re, L_pre1))
print('SVM-re')
print(diagnosis(L_re, L_pre1), 
      diagnosis(L_test0, L_pre10), 
      diagnosis(L_test1, L_pre11), 
      diagnosis(L_test2, L_pre12), 
      diagnosis(L_test3, L_pre13))


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
print('SVM: auprc and accuracy rate for the four test sets')
print(average_auprc(Prob20, L_test0), average_auprc(Prob21, L_test1), average_auprc(Prob22, L_test2), average_auprc(Prob23, L_test3))
print(accuracy_score(L_test0, L_pre20), accuracy_score(L_test1, L_pre21), accuracy_score(L_test2, L_pre22), accuracy_score(L_test3, L_pre23))
print('SVM: auprc and accuracy rate for the training set')
print(average_auprc(Prob2, L_re_smote), accuracy_score(L_re_smote, L_pre2))
print('SVM-smote')
print(diagnosis(L_re_smote, L_pre2), 
      diagnosis(L_test0, L_pre20), 
      diagnosis(L_test1, L_pre21), 
      diagnosis(L_test2, L_pre22), 
      diagnosis(L_test3, L_pre23))

clf3 = RandomForestClassifier(n_estimators=500, max_depth=50, max_leaf_nodes=50, random_state=0)
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
print('RF: auprc and accuracy rate for the four test sets')
print(average_auprc(Prob30, L_test0), average_auprc(Prob31, L_test1), average_auprc(Prob32, L_test2), average_auprc(Prob33, L_test3))
print(accuracy_score(L_test0, L_pre30), accuracy_score(L_test1, L_pre31), accuracy_score(L_test2, L_pre32), accuracy_score(L_test3, L_pre33))
print('RF: auprc and accuracy rate for the four training sets')
print(average_auprc(Prob3, L_re), accuracy_score(L_re, L_pre3))
print('RF-re')
print(diagnosis(L_re, L_pre3), 
      diagnosis(L_test0, L_pre30), 
      diagnosis(L_test1, L_pre31), 
      diagnosis(L_test2, L_pre32),  
      diagnosis(L_test3, L_pre33))

clf4 = RandomForestClassifier(n_estimators=100, max_depth=50, max_leaf_nodes=50, random_state=0)
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
print('RF: auprc and accuracy rate for the four test sets')
print(average_auprc(Prob40, L_test0), average_auprc(Prob41, L_test1), average_auprc(Prob42, L_test2), average_auprc(Prob43, L_test3))
print(accuracy_score(L_test0, L_pre40), accuracy_score(L_test1, L_pre41), accuracy_score(L_test2, L_pre42), accuracy_score(L_test3, L_pre43))
print('RF: auprc and accuracy rate for the four training sets')
print(average_auprc(Prob4, L_re_smote), accuracy_score(L_re_smote, L_pre4))
print('RF-smote')
print(diagnosis(L_re_smote, L_pre4), 
      diagnosis(L_test0, L_pre40), 
      diagnosis(L_test1, L_pre41), 
      diagnosis(L_test2, L_pre42), 
      diagnosis(L_test3, L_pre43))


clf5 = KNeighborsClassifier(n_neighbors=3)
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
print('KNN: auprc and accuracy rate for the four test sets')
print(average_auprc(Prob50, L_test0), average_auprc(Prob51, L_test1), average_auprc(Prob52, L_test2), average_auprc(Prob53, L_test3))
print(accuracy_score(L_test0, L_pre50), accuracy_score(L_test1, L_pre51), accuracy_score(L_test2, L_pre52), accuracy_score(L_test3, L_pre53))
print('KNN: auprc and accuracy rate for the four training sets')
print(average_auprc(Prob5, L_re), accuracy_score(L_re, L_pre5))
print('KNN-re')
print(diagnosis(L_re, L_pre5), 
      diagnosis(L_test0, L_pre50), 
      diagnosis(L_test1, L_pre51), 
      diagnosis(L_test2, L_pre52), 
      diagnosis(L_test3, L_pre53))

clf6 = KNeighborsClassifier(n_neighbors=3)
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
print('KNN: auprc and accuracy rate for the four test sets')
print(average_auprc(Prob60, L_test0), average_auprc(Prob61, L_test1), average_auprc(Prob62, L_test2), average_auprc(Prob63, L_test3))
print(accuracy_score(L_test0, L_pre60), accuracy_score(L_test1, L_pre61), accuracy_score(L_test2, L_pre62), accuracy_score(L_test3, L_pre63))
print('KNN: auprc and accuracy rate for the four training sets')
print(average_auprc(Prob6, L_re_smote), accuracy_score(L_re_smote, L_pre6))
print('KNN-smote')
print(diagnosis(L_re_smote, L_pre6), 
      diagnosis(L_test0, L_pre60), 
      diagnosis(L_test1, L_pre61), 
      diagnosis(L_test2, L_pre62), 
      diagnosis(L_test3, L_pre63))



