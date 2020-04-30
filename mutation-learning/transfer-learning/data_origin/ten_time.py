import numpy as np
import random

Label = np.loadtxt('m_ic50',dtype=int)
index = []
iindex = []
iiindex = []
for i in range(Label.shape[0]):
    if Label[i][0] == 1:
        index.append(i)
    if Label[i][1] == 1:
        iindex.append(i)
    if Label[i][3] == 1:
        iiindex.append(i)

T_0 = index[0:10]
T_1 = iindex[0:18]
T_3 = iiindex[0:18]
T_test = index[10:] + iindex[18:] + iiindex[18:]
np.savetxt('test_index', T_test, fmt='%d')
print(len(T_test), T_test)
print(len(T_0), T_0)
print(len(T_1), T_1)
print(len(T_3), T_3)
print('------------------')

# T_0: 1, 1, 2, 2, 2, 2
# T_1: 4, 3, 4, 3, 4
# T_3: 4, 3, 4, 3, 4

t_0_11 = T_0[0]
t_0_12 = T_0[1]
t_0_3 = T_0[2:4]
t_0_21 = T_0[4:6]
t_0_22 = T_0[6:8]
t_0_23 = T_0[8:]

t_1_41 = T_1[0:4] 
t_1_31 = T_1[4:7]
t_1_42 = T_1[7:11]
t_1_32 = T_1[11:14]
t_1_43 = T_1[14:]

t_3_41 = T_3[0:4]
t_3_31 = T_3[4:7]
t_3_42 = T_3[7:11]
t_3_32 = T_3[11:14]
t_3_43 = T_3[14:]

train_10 = [t_0_11] + t_1_41 + t_3_41
train_20 = train_10 + [t_0_12] + t_0_3 + t_1_31 + t_3_31
train_30 = train_20 + t_0_21 + t_1_42 + t_3_42
train_40 = train_30 + t_0_22 + t_1_32 + t_3_32
train_5 = train_40 + t_0_23 + t_1_43 + t_3_43

train_11 = [t_0_11] + t_1_42 + t_3_42
train_21 = train_10 + [t_0_12] + t_0_3 + t_1_32 + t_3_31
train_31 = train_20 + t_0_21 + t_1_43 + t_3_43
train_41 = train_30 + t_0_22 + t_1_31 + t_3_32

train_12 = [t_0_12] + t_1_42 + t_3_42
train_22 = train_10 + [t_0_11] + t_0_3 + t_1_32 + t_3_31
train_32 = train_20 + t_0_22 + t_1_43 + t_3_43
train_42 = train_30 + t_0_21 + t_1_31 + t_3_32

train_13 = [t_0_11] + t_1_43 + t_3_43
train_23 = train_10 + [t_0_12] + t_0_3 + t_1_31 + t_3_32
train_33 = train_20 + t_0_21 + t_1_41 + t_3_41
train_43 = train_30 + t_0_22 + t_1_32 + t_3_31

train_14 = [t_0_11] + t_1_41 + t_3_42
train_24 = train_10 + [t_0_12] + t_0_3 + t_1_32 + t_3_31
train_34 = train_20 + t_0_21 + t_1_42 + t_3_41
train_44 = train_30 + t_0_22 + t_1_31 + t_3_32

train_15 = [t_0_11] + t_1_42 + t_3_41
train_25 = train_10 + [t_0_12] + t_0_3 + t_1_31 + t_3_32
train_35 = train_20 + t_0_21 + t_1_41 + t_3_42
train_45 = train_30 + t_0_22 + t_1_32 + t_3_31

train_16 = [t_0_11] + t_1_41 + t_3_43
train_26 = train_10 + [t_0_12] + t_0_3 + t_1_31 + t_3_31
train_36 = train_20 + t_0_21 + t_1_43 + t_3_42
train_46 = train_30 + t_0_22 + t_1_32 + t_3_32

train_17 = [t_0_11] + t_1_43 + t_3_42
train_27 = train_10 + [t_0_12] + t_0_3 + t_1_31 + t_3_31
train_37 = train_20 + t_0_21 + t_1_42 + t_3_43
train_47 = train_30 + t_0_22 + t_1_32 + t_3_32

train_18 = [t_0_11] + t_1_42 + t_3_43
train_28 = train_10 + [t_0_12] + t_0_3 + t_1_31 + t_3_31
train_38 = train_20 + t_0_21 + t_1_43 + t_3_42
train_48 = train_30 + t_0_22 + t_1_32 + t_3_32

train_19 = [t_0_12] + t_1_41 + t_3_41
train_29 = train_10 + [t_0_11] + t_0_3 + t_1_32 + t_3_31
train_39 = train_20 + t_0_21 + t_1_42 + t_3_42
train_49 = train_30 + t_0_22 + t_1_31 + t_3_32

np.savetxt('train_10_index', train_10, fmt='%d')
np.savetxt('train_20_index', train_20, fmt='%d')
np.savetxt('train_30_index', train_30, fmt='%d')
np.savetxt('train_40_index', train_40, fmt='%d')
np.savetxt('train_5_index', train_5, fmt='%d')

np.savetxt('train_11_index', train_11, fmt='%d')
np.savetxt('train_21_index', train_21, fmt='%d')
np.savetxt('train_31_index', train_31, fmt='%d')
np.savetxt('train_41_index', train_41, fmt='%d')

np.savetxt('train_12_index', train_12, fmt='%d')
np.savetxt('train_22_index', train_22, fmt='%d')
np.savetxt('train_32_index', train_32, fmt='%d')
np.savetxt('train_42_index', train_42, fmt='%d')

np.savetxt('train_13_index', train_13, fmt='%d')
np.savetxt('train_23_index', train_23, fmt='%d')
np.savetxt('train_33_index', train_33, fmt='%d')
np.savetxt('train_43_index', train_43, fmt='%d')

np.savetxt('train_14_index', train_14, fmt='%d')
np.savetxt('train_24_index', train_24, fmt='%d')
np.savetxt('train_34_index', train_34, fmt='%d')
np.savetxt('train_44_index', train_44, fmt='%d')

np.savetxt('train_15_index', train_15, fmt='%d')
np.savetxt('train_25_index', train_25, fmt='%d')
np.savetxt('train_35_index', train_35, fmt='%d')
np.savetxt('train_45_index', train_45, fmt='%d')

np.savetxt('train_16_index', train_16, fmt='%d')
np.savetxt('train_26_index', train_26, fmt='%d')
np.savetxt('train_36_index', train_36, fmt='%d')
np.savetxt('train_46_index', train_46, fmt='%d')

np.savetxt('train_17_index', train_17, fmt='%d')
np.savetxt('train_27_index', train_27, fmt='%d')
np.savetxt('train_37_index', train_37, fmt='%d')
np.savetxt('train_47_index', train_47, fmt='%d')

np.savetxt('train_18_index', train_18, fmt='%d')
np.savetxt('train_28_index', train_28, fmt='%d')
np.savetxt('train_38_index', train_38, fmt='%d')
np.savetxt('train_48_index', train_48, fmt='%d')

np.savetxt('train_19_index', train_19, fmt='%d')
np.savetxt('train_29_index', train_29, fmt='%d')
np.savetxt('train_39_index', train_39, fmt='%d')
np.savetxt('train_49_index', train_49, fmt='%d')















