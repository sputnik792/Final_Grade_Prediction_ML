# %%
# The purpose of this code is to create a ANN
# The ANN is applied on aggregated features from early.csv and also Halstead features
# which can be applied on the CSEDM data to predict final exam grade on test data

# %%
# import files
import pandas as pd
import numpy as np

import random
from sklearn.preprocessing import MinMaxScaler

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf

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
newDF = pd.DataFrame(dfSubject['SubjectID'])
print(newDF)

# %%
# create y datalist for training
Y_train = dfSubject['X-Grade']
Y_train = Y_train/100
print(Y_train)

# %%
# extract features for training 

# number of problem attempted
newDF['problemAttempted'] = 0

# number of problems gotten correct eventually
newDF['NumCorrectEventually'] = 0

# total attempts
newDF['totalAttempts'] = 0

# number of problems gotten correct on first try
newDF['NumCorrectFirstTry'] = 0

for i in range(len(newDF)):
#for i in range(1):
    student = newDF['SubjectID'].iloc[i]
    studentRows = dfEarly[dfEarly['SubjectID']==student]
    #print(len(studentRows))
    newDF['problemAttempted'].iloc[i] = studentRows.shape[0]
    newDF['NumCorrectEventually'].iloc[i] = np.sum(studentRows['CorrectEventually'])
    newDF['totalAttempts'].iloc[i] = np.sum(studentRows['Attempts'])
    newDF['NumCorrectFirstTry'].iloc[i] = np.sum(studentRows['Attempts'] == 1)
print(newDF)

# %%
newDF.to_csv('data.csv')

# %%
# reading halstead for train data

halsteadDF = pd.read_csv("D:\Courses\Spring2022\CSC591 Educational data mining\Project\Dataset\F19_Release_Train_06-28-21\Fall_Train_Subject_With_Halstead.csv")
print(halsteadDF.shape)

# %%
halsteadDF = halsteadDF.drop('X-Grade',1)
print(halsteadDF.shape)


# %%
newDF = pd.merge(newDF, halsteadDF, on='SubjectID', how='left')
print(newDF.shape)
print(newDF.head(1))

# %%
# normalize the dataset 
scaler = MinMaxScaler()
X_train = newDF
for i in range(1,17):
    X_train.iloc[:,i:i+1] = scaler.fit_transform(X_train.iloc[:,i:i+1])
print(X_train)

# %%
# remove subjectID from data
X_train = X_train.drop('SubjectID',axis=1)
print(X_train)

# %%
X_train.to_csv("trainData.csv")

# %%


# %%
# reshape X_train for LSTM
X_train = X_train.to_numpy()
#X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
#print(X_train.shape)

# %%
#x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)
#print(x_train.shape)
#print(x_val.shape)

# %%
# create ANN model

model = Sequential()
# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(16,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

ad = optimizers.Adam(learning_rate=0.00001)
model.compile(loss='mean_squared_error', optimizer=ad)

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

# %%
# make prediction on train data
import math
trainPredict = model.predict(X_train)
trainScore = mean_squared_error(Y_train*100, trainPredict[:,0]*100)
print('Train Score: %.2f MSE' % (trainScore))

# 222.59 MSE

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
# reading halstead for test data

halsteadDFTest = pd.read_csv("D:\Courses\Spring2022\CSC591 Educational data mining\Project\Dataset\F19_Release_Train_06-28-21\Fall_Test_Subject_With_Halstead.csv")
print(halsteadDFTest.shape)

# %%
# create new dataframe for testing
newDFTest = pd.DataFrame(dfSubjectTest['SubjectID'])
print(newDFTest.shape)
print(newDFTest)

# %%
# extract features for testing

# number of problem attempted
newDFTest['problemAttempted'] = 0

# number of problems gotten correct eventually
newDFTest['NumCorrectEventually'] = 0

# total attempts
newDFTest['totalAttempts'] = 0

# number of problems gotten correct on first try
newDFTest['NumCorrectFirstTry'] = 0

for i in range(len(newDFTest)):
#for i in range(1):
    student = newDFTest['SubjectID'].iloc[i]
    studentRows = dfEarlyTest[dfEarlyTest['SubjectID']==student]
    #print(len(studentRows))
    newDFTest['problemAttempted'].iloc[i] = studentRows.shape[0]
    newDFTest['NumCorrectEventually'].iloc[i] = np.sum(studentRows['CorrectEventually'])
    newDFTest['totalAttempts'].iloc[i] = np.sum(studentRows['Attempts'])
    newDFTest['NumCorrectFirstTry'].iloc[i] = np.sum(studentRows['Attempts'] == 1)
print(newDFTest)

# %%
# merge the halstead data with other features
newDFTest = pd.merge(newDFTest, halsteadDFTest, on='SubjectID', how='left')
print(newDFTest.shape)
print(newDFTest.head(1))

# %%
# normalize the dataset 
scaler = MinMaxScaler()
X_test = newDFTest
for i in range(1,17):
    X_test.iloc[:,i:i+1] = scaler.fit_transform(X_test.iloc[:,i:i+1])
print(X_test)

# %%
# remove subjectID from data
X_test = X_test.drop('SubjectID',axis=1)
print(X_test)

# %%
X_test.to_csv("testData.csv")

# %%
# reshape X_test for ANN
X_test = X_test.to_numpy()
#X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
#print(X_test.shape)

# %%
# make prediction on test data
testPredict = model.predict(X_test)
testPredict = testPredict*100
print(testPredict)

# %%
dfSubjectTest['X-Grade'] = testPredict
print(dfSubjectTest)

# %%
dfSubjectTest.to_csv('output.csv')
