# import csv
# #1. process data
# csvFile = open("./data.csv",'w',newline='',encoding='utf-8')
# writer = csv.writer(csvFile)
# csvRow = []
#
# f = open("data.txt",'r',encoding='GB2312')
# for line in f:
#     csvRow = line.split(",")
#     writer.writerow(csvRow)
#
# f.close()
# csvFile.close()

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import classification_report
import matplotlib
import matplotlib.pyplot as plt

# classify different columns of data
matplotlib.use('TkAgg')
print ("\nLoading data...")
data = pd.read_csv('./data.csv', header=0)
X1 = data.iloc[:,0]
X2 = data.iloc[:,1]
X = np.column_stack((X1,X2))
Y = data.iloc[:,2]

# plot data
RedArrX = []
RedArrY = []
YellowArrX = []
YellowArrY = []
for i in range(0,len(X)):
    if Y[i] == 1:
        RedArrX.append(X1[i])
        RedArrY.append(X2[i])
    else:
        YellowArrX.append(X1[i])
        YellowArrY.append(X2[i])

s1 = plt.scatter(RedArrX,RedArrY,color='r',s=5)
s2 = plt.scatter(YellowArrX,YellowArrY,s=5)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend((s1,s2),('Actual Target Value of -1 points','Actual Target Value of +1 points') ,loc = 'best')
plt.show()

# logic regression model
import time
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=4)
train_data = np.column_stack((X_train,y_train))
# timer (calcuate the time to train LR model)
t = time.time()
# Intialize LogisticRegression
lr = linear_model.LogisticRegression(C=1, penalty='l2', tol=0.000001,class_weight= {-1:0.61,1:0.39}, solver="saga")
# Use the fit function of LogisticRegression to train the model
lr.fit(X_train, y_train)
# Predict the values
lr_y_predict = lr.predict(X_test)
lr_y_training_data = lr.predict(X_train)
# print the time of training LR model
print(f'coast:{time.time() - t:.4f}s')

# The score function of logistic regression is used to obtain
# the accurate results of the model on the test set
print ('accuracy score：', lr.score(X_test, y_test))
print (classification_report(y_test, lr_y_predict, target_names=['Benign', 'Maligant']))
# print the parameter values of the model
print(lr.coef_)
print (lr.intercept_)

# plot decision boundary function
x = np.linspace(-1, 1, 100)
y =  -x*lr.coef_[0][0]/lr.coef_[0][1] - lr.intercept_[0]/lr.coef_[0][1]
plt.plot(x, y, c='b')

# plot the actual target values of the training data
RedArrX = [] # +1 points X axis
RedArrY = [] # +1 points Y axis
YellowArrX = [] # -1 points X axis
YellowArrY = [] # -1 points Y axis

for data in train_data:
    if data[2] == 1:
        RedArrX.append(data[0])
        RedArrY.append(data[1])
    else:
        YellowArrX.append(data[0])
        YellowArrY.append(data[1])

# Plot the predictions of the training data
x_trueX = []
x_trueY= []
x_falseX = []
x_falseY = []

i = 0
for y in lr_y_training_data:
    if y == 1:
        x_trueX.append(X_train[i][0])
        x_trueY.append(X_train[i][1])
    else:
        x_falseX.append(X_train[i][0])
        x_falseY.append(X_train[i][1])
    i = i + 1

# plot predictions and actual target data of the training data
s1 = plt.scatter(x_trueX, x_trueY, color='#00CED1', s=100, marker='*')
s2 = plt.scatter(x_falseX, x_falseY, color='g', s=100,marker='*')
s3 = plt.scatter(RedArrX,RedArrY,color='#FFC0CB',s=20,marker='+')
s4 = plt.scatter(YellowArrX,YellowArrY,color = '#7FFFAA',s=20,marker='+')
plt.legend((s1,s2,s3,s4),('Predictions of -1 points','Predictions of +1 points','Actual Target Value of -1 points','Actual Target Value of +1 points') ,loc = 'best')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
