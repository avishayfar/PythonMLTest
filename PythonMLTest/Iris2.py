import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target


iris_X_train,iris_X_test, iris_y_train, iris_y_test = model_selection.train_test_split(iris_X, iris_y, test_size=0.20, random_state=7)

itemTestSize = len(iris_X_test)

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 

knsPredictY = knn.predict(iris_X_test)

rf = RandomForestClassifier (n_estimators=100)
rf.fit(iris_X_train, iris_y_train) 

rfcPredictY = rf.predict(iris_X_test)

print("iris_y_test")
print(iris_y_test)

indexes = np.arange(itemTestSize)

print("RandomForestClassifier predictY")
print(rfcPredictY)

print("KNeighborsClassifier predictY")
print(knsPredictY)

print("KNeighborsClassifier accuracy_score:")
print(accuracy_score(iris_y_test, knsPredictY))

print("RandomForestClassifier accuracy_score:")
print(accuracy_score(iris_y_test, rfcPredictY))

indexes = np.arange(itemTestSize)
lines = plt.plot(indexes, iris_y_test, 'cs-', indexes, knsPredictY, 'r*', indexes, rfcPredictY, 'b^')
plt.show()