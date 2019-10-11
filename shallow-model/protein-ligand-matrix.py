import numpy as np
import pandas as pd

A = np.load('step-one.npy')
A = A[1:3730, 0:3]
e = np.load('delete_index.npy')
A = np.delete(A, e, axis=0)      #3711 pro ligand label
a = A[:, 2]                               #3711 label
b = A[:, 0]                               #3711 pros
c = A[:, 1]                               #3711 ligands
#g = list(set(c))
#np.save('g', g)
g = np.load('g.npy')                      #2418 unique ligands
d = np.load('proteinsqe.npy')[1:29, 2]    #28 unique pros
f = A[:, 0:2]                             #3711 pro ligand pairs
train = []
test1 = []
test2 = []
test3 = []
train_lig = g[:2000]
train_pro = d[:20]
test_lig = g[2000:]
test_pro = d[20:]

for k in range(3711):
    if b[k] in train_pro and c[k] in train_lig:
        train.append(k)

for k in range(3711):
    if b[k] in train_pro and c[k] in test_lig:
        test1.append(k)

for k in range(3711):
    if b[k] in test_pro and c[k] in train_lig:
        test2.append(k)

for k in range(3711):
    if b[k] in test_pro and c[k] in test_lig:
        test3.append(k)

np.save('train_index', train)
np.save('test1_index', test1)
np.save('test2_index', test2)
np.save('test3_index', test3)
print(len(train), len(test1), len(test2), len(test3))

'''
for i in range(28):
    index = []
    for j in range(3711):
        if d[i] == B[j]:
            index.append(j)
    #print(len(index), d[i])
429 AR
245 ERA
217 ERB
0 ERRA
0 ERRG
20 FXR
789 GR
10 LRH-1
161 LXRA
245 LXRB
0 MR
268 PPARA
117 PPARB
484 PPARG
406 PR
0 PXR
5 RARA
4 RARB
2 RARG
0 RORA
66 RORG
107 RXRA
2 RXRB
28 SF-1
48 TRA
54 TRB
2 VDR
2 RXRG
'''
