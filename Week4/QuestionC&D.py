import matplotlib
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')
#load data
data = pd.read_csv('./week4_2.csv', header=None)

# process data into X and Y
X1 = data.loc[:,0]
X2 = data.loc[:,1]
X = np.column_stack((X1,X2))
Y = data.loc[:,2]

# logistic regression model
lr = linear_model.LogisticRegression(C=1, penalty='l2', tol=0.000001, solver="lbfgs",multi_class='multinomial')
poly = PolynomialFeatures(degree=1)
res = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(res, Y, test_size=0.2,random_state=42)
train_data = np.column_stack((X_train,y_train))
lr.fit(X_train, y_train)
prediction_lr = lr.predict(X_test)
# confusion matrix
print("LR MODEL CONFUSION MATRIX:\n",confusion_matrix(y_test,prediction_lr))
print(prediction_lr)
# ROC Curve
score_lr = lr.decision_function(X_test)
fpr_lr, tpr_lr, thersholds = roc_curve(y_test, score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
print(roc_auc_lr)
plt.plot(fpr_lr, tpr_lr, label="ROC (area = {0:.2f}) for LR model".format(roc_auc_lr), lw=2)

# knn model
best_model = KNeighborsClassifier(n_neighbors=51)
# read data and spilt it
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
train_data = np.column_stack((X_train,y_train))
best_model.fit(X_train, y_train)
prediction_knn = best_model.predict(X_test)
# confusion matrix
print("KNN:\n",confusion_matrix(y_test,prediction_knn))
# ROC Curve
score_knn = best_model.predict_proba(X_test)
fpr_knn, tpr_knn, thersholds = roc_curve(y_test, score_knn[:,1])
roc_auc_knn = auc(fpr_knn, tpr_knn)
print(roc_auc_knn)
plt.plot(fpr_knn, tpr_knn, label='ROC (area = {0:.2f}) for KNN'.format(roc_auc_knn), lw=2, color = 'green')

# baseline model
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier().fit(X_train, y_train)
prediction_dummy = dummy.predict(X_test)
# confusion matrix
print("DUMMY:\n",confusion_matrix(y_test,prediction_dummy))
# ROC Curve
score_dummy = dummy.predict_proba(X_test)
fpr_du, tpr_du, thersholds = roc_curve(y_test, score_dummy[:,1])
roc_auc_du = auc(fpr_du, tpr_du)
print(roc_auc_du)
plt.plot(fpr_du, tpr_du, label='ROC (area = {0:.2f}) for Dummy'.format(roc_auc_du), lw=2, color = 'red')
plt.legend(["ROC (area = {0:.2f}) for LR model".format(roc_auc_lr), 'ROC (area = {0:.2f}) for KNN'.format(roc_auc_knn),
            'ROC (area = {0:.2f}) for Dummy'.format(roc_auc_du)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()