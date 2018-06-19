import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

itemTestSize = 30
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-itemTestSize]];
iris_y_train = iris_y[indices[:-itemTestSize]]
iris_X_test  = iris_X[indices[-itemTestSize:]]
iris_y_test  = iris_y[indices[-itemTestSize:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 
KNeighborsClassifier(algorithm='auto', 
                     leaf_size=30, 
                     metric='minkowski',
                     metric_params=None, 
                     n_jobs=1, 
                     n_neighbors=5, 
                     p=2, 
                     weights='uniform')

predictY = knn.predict(iris_X_test)

print("predictY")
print(predictY)

print("iris_y_test")
print(iris_y_test)

indexes = np.arange(itemTestSize)

print("KNeighborsClassifier accuracy_score:")
print(accuracy_score(iris_y_test, predictY))

lines = plt.plot(indexes, iris_y_test, 'ro--', indexes, predictY, 'b*')
plt.show()

rf = RandomForestClassifier (n_estimators=100)
rf.fit(iris_X_train, iris_y_train) 

predictY = rf.predict(iris_X_test)

print("predictY")
print(predictY)

print("iris_y_test")
print(iris_y_test)

indexes = np.arange(itemTestSize)

print("RandomForestClassifier accuracy_score:")
print(accuracy_score(iris_y_test, predictY))

lines = plt.plot(indexes, iris_y_test, 'ro--', indexes, predictY, 'b*')
plt.show()