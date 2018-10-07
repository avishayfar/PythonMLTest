
import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


# Just to switch off pandas warning
pandas.options.mode.chained_assignment = None

url = "C:\\SeScFRD\\FinalRescan\\phase3Rescan.xlsx"

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

print("Column")
print(df.columns.values)

# Split-out validation df
array = df80.values
data_x = array[:,1:len -1]
data_y = array[:,len - 1]

x_train, x_test, y_train, y_test   =  model_selection.train_test_split (data_x, data_y, test_size = 0.2, random_state = 42)


#Start machine learning.

knn = KNeighborsClassifier()
knn.fit(x_train,y_train.astype(int))
accuracy = knn.score(x_test, y_test.astype(int))


print("Accuracy = {}%".format(accuracy * 100))

#Save the ML

joblib.dump(knn, 'C:\\SeScFRD\\SavedModels\\FraudML.pkl')


#Load the ML

knnAfterLoading = joblib.load('C:\\SeScFRD\\SavedModels\\FraudML.pkl')

#Predict

columnsOnlyX = df20.columns[1:len-1]
dfOnlyX = df20[columnsOnlyX]


df20["Predict"] = knnAfterLoading.predict(dfOnlyX)

#finalDf = df20[df20["Predict"] == 1]
#finalDf.to_excel('C:\\FRD\\FRDTransactionResult.xlsx', sheet_name='Predict')

predictlDf = df20[df20["Predict"] == 1]
rescanResultDf = df20[df20["RescanResult"] == 1]

ew = pandas.ExcelWriter("C:\\SeScFRD\\Results\\Result.xlsx")
predictlDf.to_excel(ew, sheet_name='Predict')
rescanResultDf.to_excel(ew, sheet_name='RescanResult')
ew.save() 