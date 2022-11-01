import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import classification_report
import matplotlib
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


def multiclass_logloss(actual, predicted, eps=1e-15):
    """
    Logarithmic Loss Metric
    :param actual: The array include actual target classes
    :param predicted: Classification prediction result matrix, each category has a probability
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
    for i, val in enumerate(actual):
        actual2[i, val] = 1
        actual = actual2
        clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

matplotlib.use('TkAgg')
# classify different columns of data and create two additional columns
print ("\nLoading data...")
data = pd.read_csv('./data.csv', header=0)
X1 = data.iloc[:,0]
X2 = data.iloc[:,1]
# create two additional features by adding the square of ea
X3 = X1*X1
X4 = X2*X2
X = np.column_stack((X1,X2,X3,X4))
Y = data.iloc[:,2]

# Training the Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=42)
train_data = np.column_stack((X_train,y_train))
# Intialize LogisticRegression
lr = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=0.00001,class_weight={-1:0.5,1:0.5})
# Use the fit function of LogisticRegression to train the model
lr.fit(X_train, y_train)
# Predict the value
lr_y_predict = lr.predict(X_test)
lr_y_training_data = lr.predict(X_train)
print ("logloss: %0.3f " % multiclass_logloss(y_test, lr.predict_proba(X_test)))

# The score function of logistic regression is used to obtain
# the accurate results of the model on the test set
print ('accuracy score：', lr.score(X_test, y_test))
print (classification_report(y_test, lr_y_predict, target_names=['Benign', 'Maligant']))
print(lr.coef_)
print(lr.intercept_)

# plot the decision boundary
X = np.arange(-1,1,0.001)
Y = np.arange(-1,1,0.001)
cordinates = [(x, y) for x in X for y in Y]
x_cord, y_cord = zip(*cordinates)
data = pd.DataFrame({"x":x_cord,"y":y_cord})
# Function of the decision boundary
inner = lr.coef_[0][0]*data.x + lr.coef_[0][1]*data.y + lr.coef_[0][2]*np.power(data.x,2) + lr.coef_[0][3]*np.power(data.y,2) + lr.intercept_[0]
data1 = data[np.abs(inner) < 1*10**-3]
fig,ax = plt.subplots()
ax.scatter(data1.x,data1.y,c='y',s = 2)

# plot the actual target values of the training data
RedArrX = []
RedArrY = []
YellowArrX = []
YellowArrY = []

for data in train_data:
    if data[4] == 1:
        RedArrX.append(data[0])
        RedArrY.append(data[1])
    else:
        YellowArrX.append(data[0])
        YellowArrY.append(data[1])

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


s1 = plt.scatter(x_trueX, x_trueY, color='#00CED1', s=100, marker='*')
s2 = plt.scatter(x_falseX, x_falseY, color='g', s=100,marker='*')
s3 = plt.scatter(RedArrX,RedArrY,color='#FFC0CB',s=20,marker='+')
s4 = plt.scatter(YellowArrX,YellowArrY,color = '#7FFFAA',s=20,marker='+')

plt.legend((s1,s2,s3,s4),('Actual Target Value of -1 points','Actual Target Value of +1 points','Predictions of -1 points','Predictions of +1 points') ,loc = 'best')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Baseline Predactor
clf_nb = GaussianNB() # 多项式朴素贝叶斯
clf_nb.fit(X_train, y_train)
predictions = clf_nb.predict_proba(X_test)
print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
