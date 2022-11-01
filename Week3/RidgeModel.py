import matplotlib
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')
#load data
data = pd.read_csv('./week3.csv', header=None)

# process data into X and Y
X1 = data.loc[:,0]
X2 = data.loc[:,1]
X = np.column_stack((X1,X2))
Y = data.loc[:,2]

# poly the data to power 5
poly = PolynomialFeatures(degree=5)
res = poly.fit_transform(X)

# read data and spilt it
X_train, X_test, y_train, y_test = train_test_split(res, Y, test_size=0.2,random_state=42)

# FUNCTION: get alpha value based on C
def calcAlpha(C):
    return 1/(2*C)

# Create six ridge instances by setting different alpha values
# C from 1 to 100000
ridgeC1 =Ridge(calcAlpha(1)).fit(X_train,y_train)
ridgeC10 = Ridge(calcAlpha(10)).fit(X_train,y_train)
ridgeC100 = Ridge(calcAlpha(100)).fit(X_train,y_train)
ridgeC1000 = Ridge(calcAlpha(1000)).fit(X_train,y_train)
ridgeC10000 = Ridge(calcAlpha(10000)).fit(X_train,y_train)
ridgeC100000 = Ridge(calcAlpha(100000)).fit(X_train,y_train)

# Question (i)(e)
# print the parameter information of different ridge model
# C from 1 to 100000
print("Ridge C=1")
print ("Number of features used:{}".format(np.sum(ridgeC1.coef_!=0)))
print ("Coef Parameter Values:{}".format(ridgeC1.coef_))
print ("Intercept Parameter Values:{}".format(ridgeC1.intercept_))

print("Ridge C=10")
print ("Number of features used:{}".format(np.sum(ridgeC10.coef_!=0)))
print ("Coef Parameter Values:{}".format(ridgeC10.coef_))
print ("Intercept Parameter Values:{}".format(ridgeC10.intercept_))

print("Ridge C=100")
print ("Number of features used:{}".format(np.sum(ridgeC100.coef_!=0)))
print ("Coef Parameter Values:{}".format(ridgeC100.coef_))
print ("Intercept Parameter Values:{}".format(ridgeC100.intercept_))

print("Ridge C=1000")
print ("Number of features used:{}".format(np.sum(ridgeC1000.coef_!=0)))
print ("Coef Parameter Values:{}".format(ridgeC1000.coef_))
print ("Intercept Parameter Values:{}".format(ridgeC1000.intercept_))

print("Ridge C=10000")
print ("Number of features used:{}".format(np.sum(ridgeC10000.coef_!=0)))
print ("Coef Parameter Values:{}".format(ridgeC10000.coef_))
print ("Intercept Parameter Values:{}".format(ridgeC10000.intercept_))

print("Ridge C=100000")
print ("Number of features used:{}".format(np.sum(ridgeC100000.coef_!=0)))
print ("Coef Parameter Values:{}".format(ridgeC100000.coef_))
print ("Intercept Parameter Values:{}".format(ridgeC100000.intercept_))


# Question (i)(e)
# a couple of nested for loops to create XTest
Xtest = []
grid = np.linspace(-1.5,1.5)
for i in grid:
    for j in grid:
        Xtest.append([i,j])
Xtest = np.array(Xtest)
# poly the XTest value
poly = PolynomialFeatures(degree=5)
XtestPoly = poly.fit_transform(Xtest)
# TODO: you can change differet c value based on above model to get the preValue and plot the figure,like ridgeC1,ridgeC100000
ridgePreValue = ridgeC100000.predict(XtestPoly)

# plot test predictions and training data actual target value
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
# test predictions
p1 = ax.scatter(Xtest[:,0],Xtest[:,1],ridgePreValue, ridgeC100000.predict(XtestPoly), cmap='Pastel1',s=10)
# training data actual target value
p2 = ax.scatter(X_train[:,1],X_train[:,2],y_train, c='r')
plt.title('C = 100000')
plt.xlabel('X1')
plt.ylabel('X2')
ax.set_zlabel('Y')
ax.legend([p1,p2], ['XTest Data Points','Training Data Points'], scatterpoints=1)
plt.show()

#Question (ii)(c)
mean_error=[]; std_error=[]
# Ci_range = [1, 10, 50, 100,200,500,700,1000]
# for Figure 5 range
Ci_range = np.linspace(0.01,10)
max_score = 0;
max_c = 0;
# loop of Ci_range
for Ci in Ci_range:
    # Ridge model
    model = Ridge(alpha=1/(2*Ci))
    temp=[]
    scores=[]
    # k-fold = 5
    kf = KFold(n_splits=5)
    # loop for 5-fold cross validation
    for train, test in kf.split(res):
        model.fit(res[train,0:20], Y[train])
        ypred = model.predict(res[test,0:20])
        # add mean_squared_error value
        temp.append(mean_squared_error(Y[test],ypred))
        # add cross_val_score value
        scores.append(cross_val_score(model,res[train],Y[train]))
    # get the max score of different C values
    if (max_score < np.array(scores).mean()):
        max_score = np.array(scores).mean()
        max_c = Ci
    # print the mean score for every C value
    print('C = ',Ci,'scores:', np.array(scores).mean())
    # mean and standard error
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
print('maxC = ', max_c, 'maxscores:', max_score)
# plot the mean and standard deviation use errorbar function
plt.errorbar(Ci_range,mean_error,yerr=std_error,c = 'r')
plt.xlabel('C i'); plt.ylabel('Mean square error')
plt.xlim((0,10))
plt.show()