
import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Just to switch off pandas warning
pandas.options.mode.chained_assignment = None

url = "C:\\FRD\\FRDTransactionReturnBalnce.xlsx"

df = pandas.read_excel(url)

df80 = df.sample(frac=0.8)
df20 = df.loc[~df.index.isin(df80.index)]

len = df.shape[1]

# shape
print("df")
print(df.shape)

print("df80")
print(df80.shape)

print("df20")
print(df20.shape)


# Split-out validation df
array = df80.values
data_x = array[:,1:len -1]
data_y = array[:,len - 1]

x_train, x_test, y_train, y_test   =  model_selection.train_test_split (data_x, data_y, test_size = 0.2, random_state = 42)


#Start machine learning.


columnsOnlyX = df20.columns[1:len-1]
dfOnlyX = df20[columnsOnlyX]


ew = pandas.ExcelWriter('C:\\FRD\\FRDTransactionResults5.xlsx')

#KNN
knn = KNeighborsClassifier()
knn.fit(x_train,y_train.astype(int))
accuracy = knn.score(x_test, y_test.astype(int))
print("KNeighborsClassifier Accuracy = {}%".format(accuracy * 100))

df20["Predict"] = knn.predict(dfOnlyX)

predictlDf = df20[df20["Predict"] == 1]
flagDf = df20[df20["Flag"] == 1]

predictlDf.to_excel(ew, sheet_name='KNN Predict')
flagDf.to_excel(ew, sheet_name='KNN Flag')

#RFC
rf = RandomForestClassifier (n_estimators=100)
rf.fit(x_train, y_train.astype(int))
accuracy = rf.score(x_test, y_test.astype(int))
print(" RandomForestClassifier Accuracy = {}%".format(accuracy * 100))

df20["Predict"] = rf.predict(dfOnlyX)

predictlDf = df20[df20["Predict"] == 1]
flagDf = df20[df20["Flag"] == 1]

predictlDf.to_excel(ew, sheet_name='RFC Predict')
flagDf.to_excel(ew, sheet_name='RFC Flag')

#DTC
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train.astype(int))
accuracy = dt.score(x_test, y_test.astype(int))
print(" DecisionTreeClassifier Accuracy = {}%".format(accuracy * 100))

df20["Predict"] = dt.predict(dfOnlyX)

predictlDf = df20[df20["Predict"] == 1]
flagDf = df20[df20["Flag"] == 1]

predictlDf.to_excel(ew, sheet_name='DTC Predict')
flagDf.to_excel(ew, sheet_name='DTC Flag')


ew.save() 