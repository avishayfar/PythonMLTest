418
import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Load the ML

clf = joblib.load('C:\\FRD\\FraudML.pkl')

#Predict

url = "C:\\FRD\\FRDTransaction50.xlsx"
df = pandas.read_excel(url)

columnsOnlyX = df.columns[1:18]
dfOnlyX = df[columnsOnlyX]


df["Predict"] = clf.predict(dfOnlyX)

finalDf = df[df["Predict"] == 1]

finalDf.to_excel('C:\\FRD\\FRDTransactionResult.xlsx', sheet_name='Predict')
