
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

#url = "C:\\Users\\af250128\\Desktop\\CSV\\FRDTransactionReal.xlsx"
url = "C:\\FRD\\FRDTransaction200.xlsx"

dataset = pandas.read_excel(url)

# shape
print(dataset.shape)
len = len(df.columns)

# Split-out validation dataset
array = dataset.values
data_x = array[:,1:len-1]
data_y = array[:,len-1]

x_train, x_test, y_train, y_test   =  model_selection.train_test_split (data_x, data_y, test_size = 0.2, random_state = 42)


#Start machine learning.

#knn = KNeighborsClassifier()
#knn.fit(x_train,y_train.astype(int))
#accuracy = knn.score(x_test, y_test.astype(int))

rf = RandomForestClassifier (n_estimators=100)
rf.fit(x_train, y_train.astype(int))
accuracy = rf.score(x_test, y_test.astype(int))
print("Accuracy = {}%".format(accuracy * 100))

#Save the ML

joblib.dump(rf, 'C:\\FRD\\FraudML.pkl')


#Load the ML

clf = joblib.load('C:\\FRD\\FraudML.pkl')

#Predict

url = "C:\\FRD\\FRDTransaction50.xlsx"
df = pandas.read_excel(url)

columnsOnlyX = df.columns[1:len-1]
dfOnlyX = df[columnsOnlyX]


df["Predict"] = clf.predict(dfOnlyX)

finalDf = df[df["Predict"] == 1]

finalDf.to_excel('C:\\FRD\\FRDTransactionResult.xlsx', sheet_name='Predict')
