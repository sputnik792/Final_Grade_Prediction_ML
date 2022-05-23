# The purpose of this code is to create a LSTM neural network
# The LSTM is applied on features from early.csv
# which can be applied on the CSEDM data to predict final exam grade
# Fall and Spring

# %%
# import files
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


import tensorflow as tf
import random

# %%
# fix random seed for reproducibility
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# %%
# read training data
dfSubject = pd.read_csv("D:\Courses\Spring2022\CSC591 Educational data mining\Project\Dataset\F19_Release_Train_06-28-21\Train\Data\LinkTables\Subject.csv")
print(dfSubject.shape)

# %%
# read training data
dfMainTable = pd.read_csv("D:\Courses\Spring2022\CSC591 Educational data mining\Project\Dataset\F19_Release_Train_06-28-21\Train\Data\MainTable.csv")
print(dfMainTable.shape)

# %%
# read training data
dfEarly = pd.read_csv("D:\Courses\Spring2022\CSC591 Educational data mining\Project\Dataset\F19_Release_Train_06-28-21\Train\early.csv")
print(dfEarly.shape)

# %%
# create new dataframe for training
newDF = dfEarly
newDF = newDF.drop(['AssignmentID','ProblemID'],1)
print(newDF.shape)

# %%
# converting the true/false to 0/1

for i in range(len(newDF)):
    if(newDF['CorrectEventually'].iloc[i]==True):
        newDF['CorrectEventually'].iloc[i]=1
    elif(newDF['CorrectEventually'].iloc[i]==False):
        newDF['CorrectEventually'].iloc[i]=0

    if(newDF['Label'].iloc[i]==True):
        newDF['Label'].iloc[i]=1
    elif(newDF['Label'].iloc[i]==False):
        newDF['Label'].iloc[i]=0

print(newDF)

# %%
# create y datalist for training
Y_train = dfSubject['X-Grade']
Y_train = Y_train/100
print(Y_train)
print(Y_train.shape)

# %%
# read training data for Spring

dfEarlySpring = pd.read_csv("D:/Courses/Spring2022/CSC591 Educational data mining/Project/Dataset/S19_Release_6_28_21.zip/Train/early.csv")
print(dfEarlySpring.shape)

# %%
print(len(dfEarlySpring['SubjectID'].unique()))

# %%
# create new dataframe for training
newDFSpring = dfEarlySpring
newDFSpring = newDFSpring.drop(['AssignmentID','ProblemID'],1)
print(newDFSpring.shape)

# %%
# converting the true/false to 0/1 for Spring

for i in range(len(newDFSpring)):
    if(newDFSpring['CorrectEventually'].iloc[i]==True):
        newDFSpring['CorrectEventually'].iloc[i]=1
    elif(newDFSpring['CorrectEventually'].iloc[i]==False):
        newDFSpring['CorrectEventually'].iloc[i]=0

    if(newDFSpring['Label'].iloc[i]==True):
        newDFSpring['Label'].iloc[i]=1
    elif(newDFSpring['Label'].iloc[i]==False):
        newDFSpring['Label'].iloc[i]=0

print(newDFSpring.shape)
print(newDFSpring)

# %%
# concat two data from Fall and Spring to create new training data
newDFAll = newDF.append(newDFSpring, ignore_index=True)
print(newDFAll.shape)
print(newDFAll.head(1))

# %%
newDFAll.to_csv('dataAll.csv')

# %%
# normalize the dataset between -1 and 1
#scaler = MinMaxScaler()
X_train = newDFAll
#for i in range(1,3):
#    X_train.iloc[:,i:i+1] = scaler.fit_transform(X_train.iloc[:,i:i+1])

# %%
# read Spring training data 
dfSubjectSpring = pd.read_csv("D:/Courses/Spring2022/CSC591 Educational data mining/Project/Dataset/S19_Release_6_28_21.zip/Train/Data/LinkTables/Subject.csv")
print(dfSubjectSpring.shape)

print(len(dfEarlySpring['SubjectID'].unique()))

# %%
duplicate = dfSubjectSpring[dfSubjectSpring['SubjectID'].duplicated()]
print(duplicate)


# %%
dfSubjectSpring = dfSubjectSpring.drop(75,0)
print(dfSubjectSpring.shape)

# %%
dfSubjectAll = dfSubject.append(dfSubjectSpring, ignore_index=True)
print(dfSubjectAll.shape)

# %%
# create studentList from subject.csv

studentList = dfSubjectAll['SubjectID']
print(len(studentList))
print(studentList)
studentList.to_csv("studentList.csv")


# %%
# check duplicate in studentlist
duplicate = studentList[studentList.duplicated()]
print(duplicate)


# %%
# group the per student data

data = [X_train[X_train['SubjectID'] == student] for student in studentList]

# %%
print(len(data))
print(data[367])

# %%
for i in range(len(data)):
    data[i] = data[i].drop(['SubjectID'],1)

# %%
print(data[0])
print(len(data[0]))

# %%
for i in range(len(data)):
    l = len(data[i])
    print(l)
    if(l<30):
        needed = 30-l
        print(needed)
        for j in range(needed):
            
            data[i] = data[i].append({'Attempts' : 0, 'CorrectEventually' : 0, 'Label': 0}, ignore_index = True) 
        #print(data[i])

# %%
for i in range(len(data)):
    print(data[i].shape)

# %%
print(studentList.shape)

# %%
for i in range(len(data)):
    data[i] = tf.convert_to_tensor(data[i], dtype=tf.float32)
    data[i] = np.asarray(data[i])
    #data[i] = data[i].reshape(1,data[i].shape[0],data[i].shape[1])
    print(data[i].shape)


# %%
# create y datalist for training for Spring
Y_trainSpring = dfSubjectSpring['X-Grade']
#Y_trainSpring = Y_train/100
print(Y_trainSpring.shape) 


# %%
# create new y datalist
Y_trainAll = Y_train.append(Y_trainSpring, ignore_index=True)
print(Y_trainAll.shape)
print(Y_trainAll.iloc[367])

# %%
# reshape X_train for LSTM
X_train = data
X_train = np.asarray(X_train)
#X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
#print(X_train.shape)

# %%
# divide X_train in train and val dataset
#x_train, x_val, y_train, y_val = train_test_split(X_train, Y_trainAll, test_size=0.2)
#print(x_train.shape)
#print(x_val.shape)


# %%
# create LSTM model

model = Sequential()
model.add(LSTM(20, input_shape=(30,3)))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dropout(0.1))

ad = optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=ad)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)
model.fit(X_train, Y_trainAll, epochs=100, batch_size=1, verbose=2)

# %%
#Adam:
# for learning rate 0.01, trainMSE = 199 , ValMSE = 421
# for learning rate 0.001, trainMSE = 36 , ValMSE = 390
# for learning rate 0.0001, trainMSE = 165, ValMSE = 271 
# for learing rate 0.0005, trainMSE = 77, ValMSE = 335
# for learning rate 0.00001, trainMSE = 248 , ValMSE = 330

# SGD:
# for learning rate 0.0001, trainMSE = 281, ValMSE = 349

# LSTM layer 20: 
# for learning rate 0.0001, trainMSE = 173, ValMSE = 271

# %%
# make prediction on train data
import math
trainPredict = model.predict(X_train)
trainScore = mean_squared_error(Y_trainAll*100, trainPredict[:,0]*100)
print('Train Score: %.2f MSE' % (trainScore))

# 197.05
# 184 MSE

# %%
# read test data
dfEarlyTest = pd.read_csv("D:\Courses\Spring2022\CSC591 Educational data mining\Project\Dataset\F19_Release_Test_06-28-21\Test\early.csv")
print(dfEarlyTest.shape)
print(dfEarlyTest)

# %%
# read test data
dfSubjectTest = pd.read_csv("D:\Courses\Spring2022\CSC591 Educational data mining\Project\Dataset\F19_Release_Test_06-28-21\Test\Data\LinkTables\Subject.csv")
print(dfSubjectTest.shape)
print(dfSubjectTest)

# %%
# create new dataframe for training
newDFTest = dfEarlyTest
newDFTest = newDFTest.drop(['AssignmentID','ProblemID'],1)
print(newDFTest)

# %%
# converting the true/false to 0/1

for i in range(len(newDFTest)):
    if(newDFTest['CorrectEventually'].iloc[i]==True):
        newDFTest['CorrectEventually'].iloc[i]=1
    elif(newDFTest['CorrectEventually'].iloc[i]==False):
        newDFTest['CorrectEventually'].iloc[i]=0

    if(newDFTest['Label'].iloc[i]==True):
        newDFTest['Label'].iloc[i]=1
    elif(newDFTest['Label'].iloc[i]==False):
        newDFTest['Label'].iloc[i]=0

print(newDFTest)

# %%
studentListTest = dfSubjectTest['SubjectID']
print(len(studentListTest))

# %%
# group the per student test data 

testData = [newDFTest[newDFTest['SubjectID'] == student] for student in studentListTest]

# %%
print(studentListTest.shape)

# %%
for i in range(len(testData)):
    testData[i] = testData[i].drop(['SubjectID'],1)

# %%
for i in range(len(testData)):
    l = len(testData[i])
    print(l)
    if(l<30):
        needed = 30-l
        print(needed)
        for j in range(needed):
            
            testData[i] = testData[i].append({'Attempts' : 0, 'CorrectEventually' : 0, 'Label': 0}, ignore_index = True) 
        #print(data[i])

# %%
for i in range(len(testData)):
    print(testData[i].shape)

# %%
for i in range(len(testData)):
    testData[i] = tf.convert_to_tensor(testData[i], dtype=tf.float32)
    testData[i] = np.asarray(testData[i])
    #data[i] = data[i].reshape(1,data[i].shape[0],data[i].shape[1])
    print(testData[i].shape)

# %%
X_test = testData
X_test = np.asarray(X_test)

# %%
# make prediction on train data
testPredict = model.predict(X_test)
testPredict = testPredict*100
print(testPredict)

# %%
dfSubjectTest['X-Grade'] = testPredict
print(dfSubjectTest)

# %%
dfSubjectTest.to_csv('output.csv')
