
import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Just to switch off pandas warning
pandas.options.mode.chained_assignment = None

url = "C:\\FRD\\FRDTransaction.xlsx"

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

knn = KNeighborsClassifier()
knn.fit(x_train,y_train.astype(int))
accuracy = knn.score(x_test, y_test.astype(int))

#rf = RandomForestClassifier (n_estimators=100)
#rf.fit(x_train, y_train.astype(int))
#accuracy = rf.score(x_test, y_test.astype(int))

print("Accuracy = {}%".format(accuracy * 100))

#Save the ML

#joblib.dump(rf, 'C:\\FRD\\FraudML.pkl')


#Load the ML

#clf = joblib.load('C:\\FRD\\FraudML.pkl')

#Predict

columnsOnlyX = df20.columns[1:len-1]
dfOnlyX = df20[columnsOnlyX]


df20["Predict"] = knn.predict(dfOnlyX)

#finalDf = df20[df20["Predict"] == 1]
#finalDf.to_excel('C:\\FRD\\FRDTransactionResult.xlsx', sheet_name='Predict')

predictlDf = df20[df20["Predict"] == 1]
flagDf = df20[df20["Flag"] == 1]

ew = pandas.ExcelWriter('C:\\FRD\\FRDTransactionResultsKNN.xlsx')
predictlDf.to_excel(ew, sheet_name='Predict')
flagDf.to_excel(ew, sheet_name='Flag')
ew.save() 