# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 08:44:22 2019

@author: Pias
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 22:26:23 2019

@author: Pias
"""



from __future__ import print_function
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

seq = 300 #max 300


path = r'E:\HAR_Male_Female\HAR_Male_Female\dataset\male' # use your path
all_files = glob.glob(path + "/*.csv")

X_M = pd.read_csv('E:\HAR_Male_Female\HAR_Male_Female\dataset\male\m_abubakar_180_01_1201011019.csv')
X_M = np.array(X_M)
X_M = X_M[0:seq*10, 0:6]
X_M = X_M.reshape(10, seq*6)
X_M = X_M[0:0]

for filename in all_files:
    df = pd.read_csv(filename)
    df = np.array(df)
    
    for i in range(1, seq):
        temp = df[i: , 0:6]
        for j in range(i):
            tempZ = np.zeros(6)
            tempZ = tempZ.reshape(1, tempZ.shape[0])
            
            #print(temp.shape, tempZ.shape)
            temp = np.concatenate((temp, tempZ), axis=0)
        
        #print(df.shape, temp.shape)
        df = np.column_stack((df,temp))
        #df = np.concatenate((df, temp), axis = 1)
        
    df = df[:5000]
    
    X_M = np.concatenate((X_M, df), axis=0)
    
    #print(df.shape)
    print(X_M.shape)


Y_M = []
for i in range(X_M.shape[0]):
    Y_M.append(0)

Y_M = np.array(Y_M)
Y_M = Y_M.reshape(Y_M.shape[0], 1)





x_m_train = X_M[:120000]
x_m_test = X_M[120000:]

y_m_train = Y_M[:120000]
y_m_test = Y_M[120000:]








path = r'E:\HAR_Male_Female\HAR_Male_Female\dataset\female' # use your path
all_files = glob.glob(path + "/*.csv")
X_F = pd.read_csv(r'E:\HAR_Male_Female\HAR_Male_Female\dataset\female\f_afroza_180_01_1906261832.csv')
X_F = np.array(X_F)
X_F = X_F[0:seq*10, 0:6]
X_F = X_F.reshape(10, seq*6)
X_F = X_F[0:0]

for filename in all_files:
    df = pd.read_csv(filename)
    df = np.array(df)
    
    for i in range(1, seq):
        temp = df[i: , 0:6]
        for j in range(i):
            tempZ = np.zeros(6)
            tempZ = tempZ.reshape(1, tempZ.shape[0])
            
            #print(temp.shape, tempZ.shape)
            temp = np.concatenate((temp, tempZ), axis=0)
        
        #print(df.shape, temp.shape)
        df = np.column_stack((df,temp))
        #df = np.concatenate((df, temp), axis = 1)
        
    df = df[:5000]
    
    X_F = np.concatenate((X_F, df), axis=0)
    
    #print(df.shape)
    print(X_F.shape)


Y_F = []
for i in range(X_F.shape[0]):
    Y_F.append(1)

Y_F = np.array(Y_F)
Y_F = Y_F.reshape(Y_F.shape[0], 1)





x_f_train = X_F[:80000]
x_f_test = X_F[80000:]

y_f_train = Y_F[:80000]
y_f_test = Y_F[80000:]



X_train = np.concatenate((x_m_train, x_f_train), axis=0)
X_train = np.concatenate((X_train, X_train[X_train.shape[0]-1:X_train.shape[0]]), axis=0)
np.savetxt("X_5000_seq_300_train.csv", X_train, delimiter=",")

X_test = np.concatenate((x_m_test, x_f_test), axis=0)
X_test = np.concatenate((X_test, X_test[X_test.shape[0]-1:X_test.shape[0]]), axis=0)
np.savetxt("X_5000_seq_300_test.csv", X_test, delimiter=",")

Y_train = np.concatenate((y_m_train, y_f_train), axis=0)
Y_train = np.concatenate((Y_train, Y_train[Y_train.shape[0]-1:Y_train.shape[0]]), axis=0)
np.savetxt("Y_5000_seq_300_train.csv", Y_train, delimiter=",")

Y_test = np.concatenate((y_m_test, y_f_test), axis=0)
Y_test = np.concatenate((Y_test, Y_test[Y_test.shape[0]-1:Y_test.shape[0]]), axis=0)
np.savetxt("Y_5000_seq_300_test.csv", Y_test, delimiter=",")



X = np.concatenate((X_M, X_F), axis=0)
np.savetxt("X_5000_seq_50.csv", X, delimiter=",")

Y = np.concatenate((Y_M, Y_F), axis=0)
np.savetxt("Y_5000_seq_50.csv", Y, delimiter=",")

dataset = np.concatenate((X, Y), axis=1)
np.savetxt("dataset_5000.csv", dataset, delimiter=",")


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#X_train, Y_train = shuffle(X_train, y_train)


