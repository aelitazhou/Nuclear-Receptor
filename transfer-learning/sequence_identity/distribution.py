from Bio import pairwise2
from Bio.Seq import Seq 
from Bio.SubsMat import MatrixInfo as matlist #SZ add
import time
matrix = matlist.blosum62
import os
import sys
import numpy as np


def Identity(x,y,matrix = matrix):
	l_x = len(x)
	l_y = len(y)
	X = x.upper()
	Y = y.upper()
	alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
	max_iden=0
	for i in alignments:
		same = 0
		for j in range(i[-1]):
			if i[0][j] == i[1][j] and i[0][j] != '-':
				same += 1
		iden = float(same)/float(i[-1])
		if iden > max_iden:
			max_iden = iden
	return max_iden


test_orp_2 = np.loadtxt('orphan_seq', dtype='str', delimiter=' ')
train_23 = np.loadtxt('sequence_23', dtype='str', delimiter=' ')
train_18 = np.loadtxt('sequence_18', dtype='str', delimiter=' ')
test_4 = np.loadtxt('sequence_4', dtype='str', delimiter=' ')
test_3 = np.loadtxt('sequence_3', dtype='str', delimiter=' ')
i_23 = np.arange(0, 46, 2)
i_18 = np.arange(0, 39, 2)
i_4 = np.arange(0, 8, 2)
i_3 = np.arange(0, 6, 2)
train_23 = np.delete(train_23, i_23)
train_18 = np.delete(train_18, i_18)
test_4 = np.delete(test_4, i_4)
test_3 = np.delete(test_3, i_3)

# train + both unique
score_1 = []
for i in range(18):
    for j in range(4):
        score_1.append(Identity(train_18[i], test_4[j]))
np.savetxt('18*4', score_1)
print(score_1)
print(' ')
# train + pro unique
score_2 = []
for i in range(18):
    for j in range(3):
        score_2.append(Identity(train_18[i], test_3[j]))
np.savetxt('18*3', score_2)
print(score_2)
print(' ')
# train_whole + orphan
score_3 = []
for i in range(23):
    for j in range(2):
        score_3.append(Identity(train_23[i], test_orp_2[j]))
np.savetxt('23*2', score_3)
print(score_3)
pritn(' ')
