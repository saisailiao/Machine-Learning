import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm


# classify different columns of data （the same as (a)）
matplotlib.use('TkAgg')
print ("\nLoading data...")
data = pd.read_csv('./data.csv', header=0)
X1 = data.iloc[:,0]
X2 = data.iloc[:,1]
X = np.column_stack((X1,X2))
Y = data.iloc[:,2]

import time
# Linear SVC model when C = 100
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
train_data = np.column_stack((X_train,y_train))

# actual target data of training data
RedArrX = []
RedArrY = []
YellowArrX = []
YellowArrY = []

for data in train_data:
    if data[2] == 1:
        RedArrX.append(data[0])
        RedArrY.append(data[1])
    else:
        YellowArrX.append(data[0])
        YellowArrY.append(data[1])

# timer (calcuate the time to train LR model)
t = time.time()
# Intialize Linear SVC
model = svm.LinearSVC(C=100,verbose=True)
# Use the fit function of Linear SVC to train the model
model.fit(X_train, y_train)
# Calculate the cost time
print(f'coast:{time.time() - t:.4f}s')
lr_y_training_data = model.predict(X_train)
print('score:', model.score(X_test, y_test))
print(model.coef_)
print(model.intercept_)
# plot the decision boundary
x = np.linspace(-1, 1, 100)
y =  -x*model.coef_[0][0]/model.coef_[0][1] - model.intercept_[0]/model.coef_[0][1]
print (model.coef_[0][0]/model.coef_[0][1],model.intercept_[0]/model.coef_[0][1])
plt.plot(x, y, c='b')

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

s3 = plt.scatter(x_trueX, x_trueY, color='#4B0082', s=100, marker='*')
s4 = plt.scatter(x_falseX, x_falseY, color='#FF4500', s=100, marker='*')
s1 = plt.scatter(RedArrX,RedArrY,color='#FFC0CB',s=20,marker='+')
s2 = plt.scatter(YellowArrX,YellowArrY,color = '#7FFFAA',s=20,marker='+')
plt.legend((s1,s2,s3,s4),('Actual Target Value of -1 points','Actual Target Value of +1 points','Predictions of -1 points','Predictions of +1 points') ,loc = 'best')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title('C = 100')
plt.show()

# Linear SVC model when C = 1
model1 = svm.LinearSVC(C=1,verbose=True)
model1.fit(X_train, y_train)
print('score:', model1.score(X_test, y_test))
prediction = model1.predict(X_test)
print(model1.coef_)
print(model1.intercept_)
lr_y_training_data1 = model1.predict(X_train)
x = np.linspace(-1, 1, 100)
y =  -x*model1.coef_[0][0]/model1.coef_[0][1] - model1.intercept_[0]/model1.coef_[0][1]  # 套方程公式
print (model1.coef_[0][0]/model1.coef_[0][1],model1.intercept_[0]/model1.coef_[0][1])
plt.plot(x, y, c='b')

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

s3 = plt.scatter(x_trueX, x_trueY, color='#4B0082', s=100, marker='*')
s4 = plt.scatter(x_falseX, x_falseY, color='#FF4500', s=100, marker='*')
s1 = plt.scatter(RedArrX,RedArrY,color='#FFC0CB',s=20,marker='+')
s2 = plt.scatter(YellowArrX,YellowArrY,color = '#7FFFAA',s=20,marker='+')
plt.legend((s1,s2,s3,s4),('Actual Target Value of -1 points','Actual Target Value of +1 points','Predictions of -1 points','Predictions of +1 points') ,loc = 'best')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title('C = 1')
plt.show()

model2 = svm.LinearSVC(C=0.001, verbose=True)
model2.fit(X_train, y_train)
print('score:', model2.score(X_test, y_test))
print(model2.coef_)
print(model2.intercept_)
lr_y_training_data2 = model2.predict(X_train)
# plot decision boundary
x = np.linspace(-1, 1, 100)
y =  -x*model2.coef_[0][0]/model2.coef_[0][1] - model2.intercept_[0]/model2.coef_[0][1]
print (model2.coef_[0][0]/model2.coef_[0][1],model2.intercept_[0]/model2.coef_[0][1])
plt.plot(x, y, c='b')


x_trueX = []
x_trueY= []
x_falseX = []
x_falseY = []

i = 0
for y in lr_y_training_data2:
    if y == 1:
        x_trueX.append(X_train[i][0])
        x_trueY.append(X_train[i][1])
    else:
        x_falseX.append(X_train[i][0])
        x_falseY.append(X_train[i][1])
    i = i + 1

# Plot Predictions of training data with marker '*'
s3 = plt.scatter(x_trueX, x_trueY, color='#4B0082', s=100, marker='*')
s4 = plt.scatter(x_falseX, x_falseY, color='#FF4500', s=100, marker='*')
# Plot actual target value of training data with marker '+'
s1 = plt.scatter(RedArrX,RedArrY,color='#FFC0CB',s=20,marker='+')
s2 = plt.scatter(YellowArrX,YellowArrY,color = '#7FFFAA',s=20,marker='+')
plt.legend((s1,s2,s3,s4),('Actual Target Value of -1 points','Actual Target Value of +1 points','Predictions of -1 points','Predictions of +1 points') ,loc = 'best')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title('C = 0.001')
plt.show()

