


import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

# Just to switch off pandas warning
pandas.options.mode.chained_assignment = None

url = "C:\\Users\\af250128\\Desktop\\CSV\\trainDB.csv"
names = ["Stores", "cashier", "Items Amount", "Amount", "num of Returns", "IsTbr", "isNrrLst","Fruad" ]
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)


# Split-out validation dataset
array = dataset.values
data_x = array[:,0:7]
data_y = array[:,7]

x_train, x_test, y_train, y_test   =  model_selection.train_test_split (data_x, data_y, test_size = 0.1, random_state = 42)


#Start machine learning.

rf = RandomForestClassifier (n_estimators=100)
rf.fit(x_train, y_train)
accuracy = rf.score(x_test, y_test)
print("Accuracy = {}%".format(accuracy * 100))

#Save the ML

joblib.dump(rf, 'C:\\Users\\af250128\\Desktop\\CSV\\FraudML.pkl')


#Load the ML

clf = joblib.load('C:\\Users\\af250128\\Desktop\\CSV\\FraudML.pkl') 

#Predict
url = "C:\\Users\\af250128\\Desktop\\CSV\\testDB.csv"
names = ["Stores", "cashier", "Items Amount", "Amount", "num of Returns", "IsTbr", "isNrrLst","Fruad" ]
df = pandas.read_csv(url, names=names)

columnsOnlyX = df.columns[:7]
dfOnlyX = df[columnsOnlyX]


df["Predict"] = clf.predict(dfOnlyX)

finalDf = df[df["Fruad"] == 1]

finalDf.to_excel('C:\\Users\\af250128\\Desktop\\CSV\\predict.xlsx', sheet_name='Sheet1')
