import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

m_sequence = np.loadtxt('m_sequence', dtype='str')
m_compound = np.loadtxt('m_compound', dtype='str')
m_ic50 = np.loadtxt('m_ic50', dtype='int')
 
t = np.loadtxt('test_index', dtype='int')
test_sequence = m_sequence[t]
test_compound = m_compound[t]
test_ic50 = m_ic50[t]

t_10 = np.loadtxt('train_10_index', dtype='int')
train_10_sequence = m_sequence[t_10]
train_10_compound = m_compound[t_10]
train_10_ic50 = m_ic50[t_10] 

t_20 = np.loadtxt('train_20_index', dtype='int')
train_20_sequence = m_sequence[t_20]
train_20_compound = m_compound[t_20]
train_20_ic50 = m_ic50[t_20]

t_30 = np.loadtxt('train_30_index', dtype='int')
train_30_sequence = m_sequence[t_30]
train_30_compound = m_compound[t_30]
train_30_ic50 = m_ic50[t_30]

t_40 = np.loadtxt('train_40_index', dtype='int')
train_40_sequence = m_sequence[t_40]
train_40_compound = m_compound[t_40]
train_40_ic50 = m_ic50[t_40]

t_5 = np.loadtxt('train_5_index', dtype='int')
train_5_sequence = m_sequence[t_5]
train_5_compound = m_compound[t_5]
train_5_ic50 = m_ic50[t_5]

t_11 = np.loadtxt('train_11_index', dtype='int')
train_11_sequence = m_sequence[t_11]
train_11_compound = m_compound[t_11]
train_11_ic50 = m_ic50[t_11]

t_21 = np.loadtxt('train_21_index', dtype='int')
train_21_sequence = m_sequence[t_21]
train_21_compound = m_compound[t_21]
train_21_ic50 = m_ic50[t_21]

t_31 = np.loadtxt('train_31_index', dtype='int')
train_31_sequence = m_sequence[t_31]
train_31_compound = m_compound[t_31]
train_31_ic50 = m_ic50[t_31]

t_41 = np.loadtxt('train_41_index', dtype='int')
train_41_sequence = m_sequence[t_41]
train_41_compound = m_compound[t_41]
train_41_ic50 = m_ic50[t_41]

t_12 = np.loadtxt('train_12_index', dtype='int')
train_12_sequence = m_sequence[t_12]
train_12_compound = m_compound[t_12]
train_12_ic50 = m_ic50[t_12]

t_22 = np.loadtxt('train_22_index', dtype='int')
train_22_sequence = m_sequence[t_22]
train_22_compound = m_compound[t_22]
train_22_ic50 = m_ic50[t_22]

t_32 = np.loadtxt('train_32_index', dtype='int')
train_32_sequence = m_sequence[t_32]
train_32_compound = m_compound[t_32]
train_32_ic50 = m_ic50[t_32]

t_42 = np.loadtxt('train_42_index', dtype='int')
train_42_sequence = m_sequence[t_42]
train_42_compound = m_compound[t_42]
train_42_ic50 = m_ic50[t_42]

t_13 = np.loadtxt('train_13_index', dtype='int')
train_13_sequence = m_sequence[t_13]
train_13_compound = m_compound[t_13]
train_13_ic50 = m_ic50[t_13]

t_23 = np.loadtxt('train_23_index', dtype='int')
train_23_sequence = m_sequence[t_23]
train_23_compound = m_compound[t_23]
train_23_ic50 = m_ic50[t_23]

t_33 = np.loadtxt('train_33_index', dtype='int')
train_33_sequence = m_sequence[t_33]
train_33_compound = m_compound[t_33]
train_33_ic50 = m_ic50[t_33]

t_43 = np.loadtxt('train_43_index', dtype='int')
train_43_sequence = m_sequence[t_43]
train_43_compound = m_compound[t_43]
train_43_ic50 = m_ic50[t_43]

t_14 = np.loadtxt('train_14_index', dtype='int')
train_14_sequence = m_sequence[t_14]
train_14_compound = m_compound[t_14]
train_14_ic50 = m_ic50[t_14]

t_24 = np.loadtxt('train_24_index', dtype='int')
train_24_sequence = m_sequence[t_24]
train_24_compound = m_compound[t_24]
train_24_ic50 = m_ic50[t_24]

t_34 = np.loadtxt('train_34_index', dtype='int')
train_34_sequence = m_sequence[t_34]
train_34_compound = m_compound[t_34]
train_34_ic50 = m_ic50[t_34]

t_44 = np.loadtxt('train_44_index', dtype='int')
train_44_sequence = m_sequence[t_44]
train_44_compound = m_compound[t_44]
train_44_ic50 = m_ic50[t_44]

t_15 = np.loadtxt('train_15_index', dtype='int')
train_15_sequence = m_sequence[t_15]
train_15_compound = m_compound[t_15]
train_15_ic50 = m_ic50[t_15]

t_25 = np.loadtxt('train_25_index', dtype='int')
train_25_sequence = m_sequence[t_25]
train_25_compound = m_compound[t_25]
train_25_ic50 = m_ic50[t_25]

t_35 = np.loadtxt('train_35_index', dtype='int')
train_35_sequence = m_sequence[t_35]
train_35_compound = m_compound[t_35]
train_35_ic50 = m_ic50[t_35]

t_45 = np.loadtxt('train_45_index', dtype='int')
train_45_sequence = m_sequence[t_45]
train_45_compound = m_compound[t_45]
train_45_ic50 = m_ic50[t_45]

t_16 = np.loadtxt('train_16_index', dtype='int')
train_16_sequence = m_sequence[t_16]
train_16_compound = m_compound[t_16]
train_16_ic50 = m_ic50[t_16]

t_26 = np.loadtxt('train_26_index', dtype='int')
train_26_sequence = m_sequence[t_26]
train_26_compound = m_compound[t_26]
train_26_ic50 = m_ic50[t_26]

t_36 = np.loadtxt('train_36_index', dtype='int')
train_36_sequence = m_sequence[t_36]
train_36_compound = m_compound[t_36]
train_36_ic50 = m_ic50[t_36]

t_46 = np.loadtxt('train_46_index', dtype='int')
train_46_sequence = m_sequence[t_46]
train_46_compound = m_compound[t_46]
train_46_ic50 = m_ic50[t_46]

t_17 = np.loadtxt('train_17_index', dtype='int')
train_17_sequence = m_sequence[t_17]
train_17_compound = m_compound[t_17]
train_17_ic50 = m_ic50[t_17]

t_27 = np.loadtxt('train_27_index', dtype='int')
train_27_sequence = m_sequence[t_27]
train_27_compound = m_compound[t_27]
train_27_ic50 = m_ic50[t_27]

t_37 = np.loadtxt('train_37_index', dtype='int')
train_37_sequence = m_sequence[t_37]
train_37_compound = m_compound[t_37]
train_37_ic50 = m_ic50[t_37]

t_47 = np.loadtxt('train_47_index', dtype='int')
train_47_sequence = m_sequence[t_47]
train_47_compound = m_compound[t_47]
train_47_ic50 = m_ic50[t_47]

t_18 = np.loadtxt('train_18_index', dtype='int')
train_18_sequence = m_sequence[t_18]
train_18_compound = m_compound[t_18]
train_18_ic50 = m_ic50[t_18]

t_28 = np.loadtxt('train_28_index', dtype='int')
train_28_sequence = m_sequence[t_28]
train_28_compound = m_compound[t_28]
train_28_ic50 = m_ic50[t_28]

t_38 = np.loadtxt('train_38_index', dtype='int')
train_38_sequence = m_sequence[t_38]
train_38_compound = m_compound[t_38]
train_38_ic50 = m_ic50[t_38]

t_48 = np.loadtxt('train_48_index', dtype='int')
train_48_sequence = m_sequence[t_48]
train_48_compound = m_compound[t_48]
train_48_ic50 = m_ic50[t_48]

t_19 = np.loadtxt('train_19_index', dtype='int')
train_19_sequence = m_sequence[t_19]
train_19_compound = m_compound[t_19]
train_19_ic50 = m_ic50[t_19]

t_29 = np.loadtxt('train_29_index', dtype='int')
train_29_sequence = m_sequence[t_29]
train_29_compound = m_compound[t_29]
train_29_ic50 = m_ic50[t_29]

t_39 = np.loadtxt('train_39_index', dtype='int')
train_39_sequence = m_sequence[t_39]
train_39_compound = m_compound[t_39]
train_39_ic50 = m_ic50[t_39]

t_49 = np.loadtxt('train_49_index', dtype='int')
train_49_sequence = m_sequence[t_49]
train_49_compound = m_compound[t_49]
train_49_ic50 = m_ic50[t_49]



f = open('train_10_sequence', 'w')
for i in range(len(train_10_sequence)):
    f.writelines(str(train_10_sequence[i])+'\n')
f.close
f = open('train_10_smile', 'w')
for i in range(len(train_10_compound)):
    f.writelines(str(train_10_compound[i])+'\n')
f.close
np.savetxt('train_10_ic50', train_10_ic50, fmt='%d')


f = open('train_11_sequence', 'w')
for i in range(len(train_10_sequence)):
    f.writelines(str(train_10_sequence[i])+'\n')
f.close
f = open('train_11_smile', 'w')
for i in range(len(train_11_compound)):
    f.writelines(str(train_11_compound[i])+'\n')
f.close
np.savetxt('train_11_ic50', train_11_ic50, fmt='%d')


f = open('train_12_sequence', 'w')
for i in range(len(train_12_sequence)):
    f.writelines(str(train_12_sequence[i])+'\n')
f.close
f = open('train_12_smile', 'w')
for i in range(len(train_12_compound)):
    f.writelines(str(train_12_compound[i])+'\n')
f.close
np.savetxt('train_12_ic50', train_12_ic50, fmt='%d')


f = open('train_13_sequence', 'w')
for i in range(len(train_13_sequence)):
    f.writelines(str(train_13_sequence[i])+'\n')
f.close
f = open('train_13_smile', 'w')
for i in range(len(train_13_compound)):
    f.writelines(str(train_13_compound[i])+'\n')
f.close
np.savetxt('train_13_ic50', train_13_ic50, fmt='%d')


f = open('train_14_sequence', 'w')
for i in range(len(train_14_sequence)):
    f.writelines(str(train_14_sequence[i])+'\n')
f.close
f = open('train_14_smile', 'w')
for i in range(len(train_14_compound)):
    f.writelines(str(train_14_compound[i])+'\n')
f.close
np.savetxt('train_14_ic50', train_14_ic50, fmt='%d')


f = open('train_15_sequence', 'w')
for i in range(len(train_15_sequence)):
    f.writelines(str(train_15_sequence[i])+'\n')
f.close
f = open('train_15_smile', 'w')
for i in range(len(train_15_compound)):
    f.writelines(str(train_15_compound[i])+'\n')
f.close
np.savetxt('train_15_ic50', train_15_ic50, fmt='%d')


f = open('train_16_sequence', 'w')
for i in range(len(train_16_sequence)):
    f.writelines(str(train_16_sequence[i])+'\n')
f.close
f = open('train_16_smile', 'w')
for i in range(len(train_16_compound)):
    f.writelines(str(train_16_compound[i])+'\n')
f.close
np.savetxt('train_16_ic50', train_16_ic50, fmt='%d')


f = open('train_17_sequence', 'w')
for i in range(len(train_17_sequence)):
    f.writelines(str(train_17_sequence[i])+'\n')
f.close
f = open('train_17_smile', 'w')
for i in range(len(train_17_compound)):
    f.writelines(str(train_17_compound[i])+'\n')
f.close
np.savetxt('train_17_ic50', train_17_ic50, fmt='%d')


f = open('train_18_sequence', 'w')
for i in range(len(train_18_sequence)):
    f.writelines(str(train_18_sequence[i])+'\n')
f.close
f = open('train_18_smile', 'w')
for i in range(len(train_18_compound)):
    f.writelines(str(train_18_compound[i])+'\n')
f.close
np.savetxt('train_18_ic50', train_18_ic50, fmt='%d')


f = open('train_19_sequence', 'w')
for i in range(len(train_19_sequence)):
    f.writelines(str(train_19_sequence[i])+'\n')
f.close
f = open('train_19_smile', 'w')
for i in range(len(train_19_compound)):
    f.writelines(str(train_19_compound[i])+'\n')
f.close
np.savetxt('train_19_ic50', train_19_ic50, fmt='%d')


f = open('train_20_sequence', 'w')
for i in range(len(train_20_sequence)):
    f.writelines(str(train_20_sequence[i])+'\n')
f.close
f = open('train_20_smile', 'w')
for i in range(len(train_20_compound)):
    f.writelines(str(train_20_compound[i])+'\n')
f.close
np.savetxt('train_20_ic50', train_20_ic50, fmt='%d')


f = open('train_21_sequence', 'w')
for i in range(len(train_21_sequence)):
    f.writelines(str(train_21_sequence[i])+'\n')
f.close
f = open('train_21_smile', 'w')
for i in range(len(train_21_compound)):
    f.writelines(str(train_21_compound[i])+'\n')
f.close
np.savetxt('train_21_ic50', train_21_ic50, fmt='%d')


f = open('train_22_sequence', 'w')
for i in range(len(train_22_sequence)):
    f.writelines(str(train_22_sequence[i])+'\n')
f.close
f = open('train_22_smile', 'w')
for i in range(len(train_22_compound)):
    f.writelines(str(train_22_compound[i])+'\n')
f.close
np.savetxt('train_22_ic50', train_22_ic50, fmt='%d')



f = open('train_23_sequence', 'w')
for i in range(len(train_23_sequence)):
    f.writelines(str(train_23_sequence[i])+'\n')
f.close
f = open('train_23_smile', 'w')
for i in range(len(train_23_compound)):
    f.writelines(str(train_23_compound[i])+'\n')
f.close
np.savetxt('train_23_ic50', train_23_ic50, fmt='%d')


f = open('train_24_sequence', 'w')
for i in range(len(train_24_sequence)):
    f.writelines(str(train_24_sequence[i])+'\n')
f.close
f = open('train_24_smile', 'w')
for i in range(len(train_24_compound)):
    f.writelines(str(train_24_compound[i])+'\n')
f.close
np.savetxt('train_24_ic50', train_24_ic50, fmt='%d')


f = open('train_25_sequence', 'w')
for i in range(len(train_25_sequence)):
    f.writelines(str(train_25_sequence[i])+'\n')
f.close
f = open('train_25_smile', 'w')
for i in range(len(train_25_compound)):
    f.writelines(str(train_25_compound[i])+'\n')
f.close
np.savetxt('train_25_ic50', train_25_ic50, fmt='%d')


f = open('train_26_sequence', 'w')
for i in range(len(train_26_sequence)):
    f.writelines(str(train_26_sequence[i])+'\n')
f.close
f = open('train_26_smile', 'w')
for i in range(len(train_26_compound)):
    f.writelines(str(train_26_compound[i])+'\n')
f.close
np.savetxt('train_26_ic50', train_26_ic50, fmt='%d')


f = open('train_27_sequence', 'w')
for i in range(len(train_27_sequence)):
    f.writelines(str(train_27_sequence[i])+'\n')
f.close
f = open('train_27_smile', 'w')
for i in range(len(train_27_compound)):
    f.writelines(str(train_27_compound[i])+'\n')
f.close
np.savetxt('train_27_ic50', train_27_ic50, fmt='%d')


f = open('train_28_sequence', 'w')
for i in range(len(train_28_sequence)):
    f.writelines(str(train_28_sequence[i])+'\n')
f.close
f = open('train_28_smile', 'w')
for i in range(len(train_28_compound)):
    f.writelines(str(train_28_compound[i])+'\n')
f.close
np.savetxt('train_28_ic50', train_28_ic50, fmt='%d')


f = open('train_29_sequence', 'w')
for i in range(len(train_29_sequence)):
    f.writelines(str(train_29_sequence[i])+'\n')
f.close
f = open('train_29_smile', 'w')
for i in range(len(train_29_compound)):
    f.writelines(str(train_29_compound[i])+'\n')
f.close
np.savetxt('train_29_ic50', train_29_ic50, fmt='%d')


f = open('train_30_sequence', 'w')
for i in range(len(train_30_sequence)):
    f.writelines(str(train_30_sequence[i])+'\n')
f.close
f = open('train_30_smile', 'w')
for i in range(len(train_30_compound)):
    f.writelines(str(train_30_compound[i])+'\n')
f.close
np.savetxt('train_30_ic50', train_30_ic50, fmt='%d')


f = open('train_31_sequence', 'w')
for i in range(len(train_31_sequence)):
    f.writelines(str(train_31_sequence[i])+'\n')
f.close
f = open('train_31_smile', 'w')
for i in range(len(train_31_compound)):
    f.writelines(str(train_31_compound[i])+'\n')
f.close
np.savetxt('train_31_ic50', train_31_ic50, fmt='%d')


f = open('train_32_sequence', 'w')
for i in range(len(train_32_sequence)):
    f.writelines(str(train_32_sequence[i])+'\n')
f.close
f = open('train_32_smile', 'w')
for i in range(len(train_32_compound)):
    f.writelines(str(train_32_compound[i])+'\n')
f.close
np.savetxt('train_32_ic50', train_32_ic50, fmt='%d')


f = open('train_33_sequence', 'w')
for i in range(len(train_33_sequence)):
    f.writelines(str(train_33_sequence[i])+'\n')
f.close
f = open('train_33_smile', 'w')
for i in range(len(train_33_compound)):
    f.writelines(str(train_33_compound[i])+'\n')
f.close
np.savetxt('train_33_ic50', train_33_ic50, fmt='%d')


f = open('train_34_sequence', 'w')
for i in range(len(train_34_sequence)):
    f.writelines(str(train_34_sequence[i])+'\n')
f.close
f = open('train_34_smile', 'w')
for i in range(len(train_34_compound)):
    f.writelines(str(train_34_compound[i])+'\n')
f.close
np.savetxt('train_34_ic50', train_34_ic50, fmt='%d')


f = open('train_35_sequence', 'w')
for i in range(len(train_35_sequence)):
    f.writelines(str(train_35_sequence[i])+'\n')
f.close
f = open('train_35_smile', 'w')
for i in range(len(train_35_compound)):
    f.writelines(str(train_35_compound[i])+'\n')
f.close
np.savetxt('train_35_ic50', train_35_ic50, fmt='%d')


f = open('train_36_sequence', 'w')
for i in range(len(train_36_sequence)):
    f.writelines(str(train_36_sequence[i])+'\n')
f.close
f = open('train_36_smile', 'w')
for i in range(len(train_36_compound)):
    f.writelines(str(train_36_compound[i])+'\n')
f.close
np.savetxt('train_36_ic50', train_36_ic50, fmt='%d')


f = open('train_37_sequence', 'w')
for i in range(len(train_37_sequence)):
    f.writelines(str(train_37_sequence[i])+'\n')
f.close
f = open('train_37_smile', 'w')
for i in range(len(train_37_compound)):
    f.writelines(str(train_37_compound[i])+'\n')
f.close
np.savetxt('train_37_ic50', train_37_ic50, fmt='%d')


f = open('train_38_sequence', 'w')
for i in range(len(train_38_sequence)):
    f.writelines(str(train_38_sequence[i])+'\n')
f.close
f = open('train_38_smile', 'w')
for i in range(len(train_38_compound)):
    f.writelines(str(train_38_compound[i])+'\n')
f.close
np.savetxt('train_38_ic50', train_38_ic50, fmt='%d')


f = open('train_39_sequence', 'w')
for i in range(len(train_39_sequence)):
    f.writelines(str(train_39_sequence[i])+'\n')
f.close
f = open('train_39_smile', 'w')
for i in range(len(train_39_compound)):
    f.writelines(str(train_39_compound[i])+'\n')
f.close
np.savetxt('train_39_ic50', train_39_ic50, fmt='%d')


f = open('train_40_sequence', 'w')
for i in range(len(train_40_sequence)):
    f.writelines(str(train_40_sequence[i])+'\n')
f.close
f = open('train_40_smile', 'w')
for i in range(len(train_40_compound)):
    f.writelines(str(train_40_compound[i])+'\n')
f.close
np.savetxt('train_40_ic50', train_40_ic50, fmt='%d')


f = open('train_41_sequence', 'w')
for i in range(len(train_41_sequence)):
    f.writelines(str(train_41_sequence[i])+'\n')
f.close
f = open('train_41_smile', 'w')
for i in range(len(train_41_compound)):
    f.writelines(str(train_41_compound[i])+'\n')
f.close
np.savetxt('train_41_ic50', train_41_ic50, fmt='%d')


f = open('train_42_sequence', 'w')
for i in range(len(train_42_sequence)):
    f.writelines(str(train_42_sequence[i])+'\n')
f.close
f = open('train_42_smile', 'w')
for i in range(len(train_42_compound)):
    f.writelines(str(train_42_compound[i])+'\n')
f.close
np.savetxt('train_42_ic50', train_42_ic50, fmt='%d')


f = open('train_43_sequence', 'w')
for i in range(len(train_43_sequence)):
    f.writelines(str(train_43_sequence[i])+'\n')
f.close
f = open('train_43_smile', 'w')
for i in range(len(train_43_compound)):
    f.writelines(str(train_43_compound[i])+'\n')
f.close
np.savetxt('train_43_ic50', train_43_ic50, fmt='%d')


f = open('train_44_sequence', 'w')
for i in range(len(train_44_sequence)):
    f.writelines(str(train_44_sequence[i])+'\n')
f.close
f = open('train_44_smile', 'w')
for i in range(len(train_44_compound)):
    f.writelines(str(train_44_compound[i])+'\n')
f.close
np.savetxt('train_44_ic50', train_44_ic50, fmt='%d')


f = open('train_45_sequence', 'w')
for i in range(len(train_45_sequence)):
    f.writelines(str(train_45_sequence[i])+'\n')
f.close
f = open('train_45_smile', 'w')
for i in range(len(train_45_compound)):
    f.writelines(str(train_45_compound[i])+'\n')
f.close
np.savetxt('train_45_ic50', train_45_ic50, fmt='%d')


f = open('train_46_sequence', 'w')
for i in range(len(train_46_sequence)):
    f.writelines(str(train_46_sequence[i])+'\n')
f.close
f = open('train_46_smile', 'w')
for i in range(len(train_46_compound)):
    f.writelines(str(train_46_compound[i])+'\n')
f.close
np.savetxt('train_46_ic50', train_46_ic50, fmt='%d')


f = open('train_47_sequence', 'w')
for i in range(len(train_47_sequence)):
    f.writelines(str(train_47_sequence[i])+'\n')
f.close
f = open('train_47_smile', 'w')
for i in range(len(train_47_compound)):
    f.writelines(str(train_47_compound[i])+'\n')
f.close
np.savetxt('train_47_ic50', train_47_ic50, fmt='%d')


f = open('train_48_sequence', 'w')
for i in range(len(train_48_sequence)):
    f.writelines(str(train_48_sequence[i])+'\n')
f.close
f = open('train_48_smile', 'w')
for i in range(len(train_48_compound)):
    f.writelines(str(train_48_compound[i])+'\n')
f.close
np.savetxt('train_48_ic50', train_48_ic50, fmt='%d')


f = open('train_49_sequence', 'w')
for i in range(len(train_49_sequence)):
    f.writelines(str(train_49_sequence[i])+'\n')
f.close
f = open('train_49_smile', 'w')
for i in range(len(train_49_compound)):
    f.writelines(str(train_49_compound[i])+'\n')
f.close
np.savetxt('train_49_ic50', train_49_ic50, fmt='%d')


f = open('train_5_sequence', 'w')
for i in range(len(train_5_sequence)):
    f.writelines(str(train_5_sequence[i])+'\n')
f.close
f = open('train_5_smile', 'w')
for i in range(len(train_5_compound)):
    f.writelines(str(train_5_compound[i])+'\n')
f.close
np.savetxt('train_5_ic50', train_5_ic50, fmt='%d')


f = open('test_sequence', 'w')
for i in range(len(test_sequence)):
    f.writelines(str(test_sequence[i])+'\n')
f.close
f = open('test_smile', 'w')
for i in range(len(test_compound)):
    f.writelines(str(test_compound[i])+'\n')
f.close
np.savetxt('test_ic50', test_ic50, fmt='%d')

