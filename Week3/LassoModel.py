import matplotlib
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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

#Question (i)(a)
#define axis
fig = plt.figure()
# define 3d plot
ax = fig.add_subplot(111,projection='3d')
p = ax.scatter3D(X1,X2,Y, cmap='Blues')
plt.xlabel('X1')
plt.ylabel('X2')
ax.set_zlabel('Y')
ax.legend([p], ['Downloaded Data Points'], loc='best', scatterpoints=1)
plt.show()

# FUNCTION: get alpha value based on C
def calcAlpha(C):
    return 1/(2*C)

# Create six lasso instances by setting different alpha values
# C from 1 to 100000
lassoC1 =Lasso(calcAlpha(1)).fit(X_train,y_train)
lassoC10 = Lasso(calcAlpha(10)).fit(X_train,y_train)
lassoC100 = Lasso(calcAlpha(100)).fit(X_train,y_train)
lassoC1000 = Lasso(calcAlpha(1000)).fit(X_train,y_train)
lassoC10000 = Lasso(calcAlpha(10000)).fit(X_train,y_train)
lassoC100000 = Lasso(calcAlpha(100000)).fit(X_train,y_train)

# Question (i)(b)
# print the parameter information of different lasso model
# C from 1 to 100000
print("Lasso C=1")
print ("training data score:{:.2f}".format(lassoC1.score(X_train,y_train)))
print ("test data score:{:.2f}".format(lassoC1.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lassoC1.coef_!=0)))
print ("Coef Parameter Values:{}".format(lassoC1.coef_))
print ("Intercept Parameter Values:{}".format(lassoC1.intercept_))

print("Lasso C=10")
print ("Used feature:{}".format(np.sum(lassoC10.coef_!=0)))
print ("Coef Parameter Values:{}".format(lassoC10.coef_))
print ("Intercept Parameter Values:{}".format(lassoC10.intercept_))

print("Lasso C=100")
print ("Used feature:{}".format(np.sum(lassoC100.coef_!=0)))
print ("Coef Parameter Values:{}".format(lassoC100.coef_))
print ("Intercept Parameter Values:{}".format(lassoC100.intercept_))

print("Lasso C=1000")
print ("Used feature:{}".format(np.sum(lassoC1000.coef_!=0)))
print ("Coef Parameter Values:{}".format(lassoC1000.coef_))
print ("Intercept Parameter Values:{}".format(lassoC1000.intercept_))

print("Lasso C=10000")
print ("Used feature:{}".format(np.sum(lassoC10000.coef_!=0)))
print ("Coef Parameter Values:{}".format(lassoC10000.coef_))
print ("Intercept Parameter Values:{}".format(lassoC10000.intercept_))

print("Lasso C=100000")
print ("Used feature:{}".format(np.sum(lassoC100000.coef_!=0)))
print ("Coef Parameter Values:{}".format(lassoC100000.coef_))
print ("Intercept Parameter Values:{}".format(lassoC100000.intercept_))

# Question (i)(c)
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
# TODO: you can change differet c value based on above model to get the preValue and plot the figure,like lassoC1,lassoC100000
lassoPreValue = lassoC100000.predict(XtestPoly)

# plot test predictions and training data actual target value
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
# test predictions
p1 = ax.scatter(Xtest[:,0],Xtest[:,1],lassoPreValue, cmap='Pastel1',s=10)
# training data actual target value
p2 = ax.scatter(X_train[:,1],X_train[:,2],y_train, c='r')
plt.title('C = 10000')
plt.xlabel('X1')
plt.ylabel('X2')
ax.set_zlabel('Y')
ax.legend([p1,p2], ['XTest Data Points','Training Data Points'], scatterpoints=1)
plt.show()

#Question (ii)(a)(b)
mean_error=[]; std_error=[]
# Ci_range = [1, 10, 50, 100,200,500,700,1500]
# for Figure 5 range
Ci_range = np.linspace(100,200)
# the best score and C value
max_score = 0;
max_c = 0;
# loop of Ci_range
for Ci in Ci_range:
    # lasso model
    model = Lasso(alpha=1/(2*Ci))
    temp=[]
    scores=[]
    # k-fold = 5
    kf = KFold(n_splits=5)
    # loop for 5-fold cross validation
    for train, test in kf.split(res):
        model.fit(res[train,0:20], Y[train])
        y_prediction = model.predict(res[test,0:20])
        # add mean_squared_error value
        temp.append(mean_squared_error(Y[test],y_prediction))
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
plt.errorbar(Ci_range,mean_error,yerr=std_error)
plt.xlabel('C i'); plt.ylabel('Mean square error')
plt.xlim((100,201))
plt.show()