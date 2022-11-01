import matplotlib
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')
#load data
# first dataset: week_1.data
# second dataset: week_2.data
data = pd.read_csv('./week4_2.csv', header=None)

# process data into X and Y
X1 = data.loc[:,0]
X2 = data.loc[:,1]
X = np.column_stack((X1,X2))
Y = data.loc[:,2]

# plot original data
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
print(len(RedArrX),len(YellowArrX))
s1 = plt.scatter(RedArrX,RedArrY,color='r',s=5)
s2 = plt.scatter(YellowArrX,YellowArrY,s=5)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend((s1,s2),('Actual Target Value of +1 points','Actual Target Value of -1 points') ,loc = 'best')
# plt.show()

# poly the data to power n
# large range
Ci_range = [0.1,1,10,100,1000]
# small range
Ci_range = np.linspace(0.1,100)
# variable
total_mean_error = []
total_std_error = []
Ci_all = []
degree_all = []
lowest_mse_index = 0
# 5 FOLD CROSS VALIDATION
for ci in Ci_range:
    best_degree = 0
    mean_error = []
    std_error = []
    lr = linear_model.LogisticRegression(C=ci, penalty='l2', tol=0.000001, solver="lbfgs",multi_class='multinomial')
    for i in range(1,10):
        poly = PolynomialFeatures(degree=i)
        res = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(res, Y, test_size=0.2,random_state=42)
        lr.fit(X_train, y_train)
        scores = []
        temp = []
        kf = KFold(n_splits=5)
        # loop for 5-fold cross validation
        for train, test in kf.split(res):
            lr.fit(res[train], Y[train])
            y_prediction = lr.predict(res[test])
            # add cross_val_score value
            scores.append(cross_val_score(lr,X_train,y_train))
            # add mean_squared_error value
            temp.append(mean_squared_error(Y[test], y_prediction))
        # mean and standard error
        total_mean_error.append(np.array(temp).mean())
        total_std_error.append(np.array(temp).std())
        Ci_all.append(ci)
        degree_all.append(i)
        if (total_mean_error[lowest_mse_index] > total_mean_error[-1]):
            lowest_mse_index = len(total_mean_error) - 1

print("lowest mse:",total_mean_error[lowest_mse_index],"Ci:",Ci_all[lowest_mse_index],"degree:",degree_all[lowest_mse_index])
# plot the mean and standard deviation use errorbar function
ax = plt.figure().add_subplot(projection='3d')
# plt.errorbar(range(1,17), mean_error, yerr=std_error)
ax.errorbar(Ci_all, degree_all, total_mean_error,yerr = total_std_error, zuplims=0.01, zlolims=0.01)
plt.ylabel('degree range')
plt.xlabel('Ci')
ax.set_zlabel("Mean Square Error")
plt.title("MSE Value Based on Various C and Degree q")
ax.plot(Ci_all[lowest_mse_index],degree_all[lowest_mse_index],total_mean_error[lowest_mse_index],'o',color = 'r')
# plt.xlim((1, 16))
plt.show()

# TRAIN THE BEST PARAMETER MODEL
# FOR FIRST DATASET C = 4 DEGREE = 2
# FOR SECOND DATASET C = 1 DEGREE = 1
lr = linear_model.LogisticRegression(C=1, penalty='l2', tol=0.000001, solver="lbfgs",multi_class='multinomial')
poly = PolynomialFeatures(degree=1)
res = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(res, Y, test_size=0.2,random_state=42)
train_data = np.column_stack((X_train,y_train))
lr.fit(X_train, y_train)
print("coef:",lr.coef_,"intercept:",lr.intercept_)

prediction_lr = lr.predict(X_train)
print(prediction_lr)

# plot the actual target values of the training data
RedArrX = []  # +1 points X axis
RedArrY = []  # +1 points Y axis
YellowArrX = []  # -1 points X axis
YellowArrY = []  # -1 points Y axis

for data in train_data:
    if data[-1] == 1:
        RedArrX.append(data[1])
        RedArrY.append(data[2])
    else:
        YellowArrX.append(data[1])
        YellowArrY.append(data[2])

# FUNCTION: PLOT TRAINING DATA AND PREDICTIONS
def plotTrainandPredictions():

    # Plot the predictions of the training data
    x_trueX = []
    x_trueY = []
    x_falseX = []
    x_falseY = []

    i = 0

    for y in prediction_lr:
        if y == 1:
            x_trueX.append(X_train[i][1])
            x_trueY.append(X_train[i][2])
        else:
            x_falseX.append(X_train[i][1])
            x_falseY.append(X_train[i][2])
        i = i + 1

    # plot predictions and actual target data of the training data
    s1 = plt.scatter(x_trueX, x_trueY,color='#00CED1', s=100, marker='*')
    s2 = plt.scatter(x_falseX, x_falseY, color='g', s=100, marker='*')
    s3 = plt.scatter(RedArrX, RedArrY, color='#FFC0CB', s=20, marker='+')
    s4 = plt.scatter(YellowArrX, YellowArrY, color='#7FFFAA', s=20, marker='+')
    plt.legend((s1, s2, s3, s4), ('Predictions of +1 points', 'Predictions of -1 points', 'Actual Target Value of +1 points',
    'Actual Target Value of -1 points'), loc='best')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


plotTrainandPredictions()

