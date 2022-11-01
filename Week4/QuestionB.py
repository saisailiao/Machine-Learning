import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold  # 用于k折交叉验证
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')
#load data
# first dataset: week_1.data
# second dataset: week_2.data
data = pd.read_csv('./week4_1.csv', header=None)

# process data into X and Y
X1 = data.loc[:,0]
X2 = data.loc[:,1]
X = np.column_stack((X1,X2))
Y = data.loc[:,2]

k_fold = 5
# set the kfold value, also set shuffle = True so that the result of each partition is different
kf = KFold(n_splits=k_fold, random_state=2001, shuffle=True)

# save the best k value
best_k = 1
best_score = 0

# mean error & std_error
mean_error=[]; std_error=[]

# set an array which contains the potential best k value
# I set the potential k range from 1 to 100
k_num = [2, 10, 30, 50, 100, 200, 300,500,1000]
average_score = []
lowest_mse = 100
# large range of k_num
for k in range(2,150):
# small range of k_num
# for k in range(10,30):
    curScore = 0
    temp = []
    # Each fold of training and calculation accuracy
    for train, test in kf.split(X):
        # config the number of neighors
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X[train], Y[train])
        curScore = curScore + model.score(X[train], Y[train])
        y_prediction = model.predict(X[test]);
        # add mean_squared_error value
        temp.append(mean_squared_error(Y[test], y_prediction))

    # average score of the every fold result
    avg_score = curScore / k_fold
    average_score.append(avg_score)
    # mean and standard error
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
    # get the lowest MSE value
    if lowest_mse > mean_error[-1]:
        lowest_mse = mean_error[-1]
        best_k = k


print("final best n_neighbors:%d" % best_k, "final lowest mse：%.3f" % lowest_mse)

# plot the mean and standard deviation use errorbar function
#large range
plt.errorbar(range(2,150),mean_error,yerr=std_error)
print(mean_error)
# small range
# plt.errorbar(range(10,30),mean_error,yerr=std_error,ecolor='r')
plt.xlabel('K-Neighbour')
plt.ylabel('Mean square error')
plt.title("KNN - 5 fold CV")
plt.xlim((2,155))
# plt.xlim((10,31))
plt.show()

# use the best k value to train knn model
# FOR FIRST DATASET K = 12
# FOR SECOND DATASET K = 51
best_model = KNeighborsClassifier(n_neighbors=12)
# read data and spilt it
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
train_data = np.column_stack((X_train,y_train))
best_model.fit(X_train, y_train)
prediction_knn = best_model.predict(X_train)


# plot the actual target values of the training data
RedArrX = []  # +1 points X axis
RedArrY = []  # +1 points Y axis
YellowArrX = []  # -1 points X axis
YellowArrY = []  # -1 points Y axis

for data in train_data:
    if data[2] == 1:
        RedArrX.append(data[0])
        RedArrY.append(data[1])
    else:
        YellowArrX.append(data[0])
        YellowArrY.append(data[1])

# FUNCTION: PLOT TRAINING DATA AND PREDICTIONS
def plotTrainandPredictions():

    # Plot the predictions of the training data
    x_trueX = []
    x_trueY = []
    x_falseX = []
    x_falseY = []

    i = 0
    for y in prediction_knn:
        if y == 1:
            x_trueX.append(X_train[i][0])
            x_trueY.append(X_train[i][1])
        else:
            x_falseX.append(X_train[i][0])
            x_falseY.append(X_train[i][1])
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

























