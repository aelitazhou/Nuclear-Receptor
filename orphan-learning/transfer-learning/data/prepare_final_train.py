import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


P = np.loadtxt('train_10_sps', dtype='str')
C = []
for line in open('train_10_smile', 'r'):
    C.append(line)
C = np.array(C) 
Label = np.loadtxt('train_10_ic50', dtype='int')
L = []
for i in range(Label.shape[0]):
    if Label[i, 0] == 1:
        L.append(0)
    else:
        L.append(1)
L = np.array(L)
print('----------------')
print('number of sps, cansmile, label')
print(P.shape, C.shape, L.shape)

# oversampling 
lb = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
ros = RandomOverSampler(random_state=0)
# train_final
F = np.column_stack((P, C))
F_re, L_re = ros.fit_sample(F, L)
P_re = F_re[:,0]
C_re = F_re[:,1]
print('----------------')
print('shape of sps, cansmile, label of train_whole after oversampling')
print(P_re.shape, C_re.shape, L_re.shape)

P_re = np.concatenate((P_re, P_re[2:8]))
C_re = np.concatenate((C_re, C_re[2:8]))
L_re = np.concatenate((L_re, L_re[2:8]))


f = open('train_10final_sps', 'w')
for i in range(len(P_re)):
    f.writelines(str(P_re[i])+'\n')
f.close
f = open('train_10final_smile', 'w')
for i in range(len(C_re)):
    f.writelines(str(C_re[i]))
f.close
np.savetxt('train_10final_ic50', L_re, fmt='%d')

