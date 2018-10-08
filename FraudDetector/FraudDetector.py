import pandas as pd
import numpy as np
from sqlalchemy import *
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


def NormlizeColumnsValue (df80, df20, namesOfColumns, threshold):
    for columnName in namesOfColumns:
        basicStd  = df80[columnName].std()
        basicMean = df80[columnName].mean()
        maxValue = basicStd * threshold
        minValue = (-1)*maxValue
        df80[columnName] = df80[columnName].pipe(lambda x: (x-basicMean) / x.std())
        df80[columnName][df80[columnName] > maxValue] = maxValue
        df80[columnName][df80[columnName] < minValue] = minValue
        df20[columnName] = df20[columnName].pipe(lambda x: (x-basicMean) / x.std())
        df20[columnName][df20[columnName] > maxValue] = maxValue
        df20[columnName][df20[columnName] < minValue] = minValue
       


namesOfColumn4Learning = ['TrxTotalNzd','VoidedItemsNzd','ValueVoidedLinesNzd','EarlyMorning','lunch','QuantitiyItemsNzd']
namesOfColumn4Normliaze  =['TrxTotalNzd','VoidedItemsNzd','ValueVoidedLinesNzd','QuantitiyItemsNzd']
nameOfRescanResult = 'RescanResult'
threshold = 3 

engine = create_engine('postgresql://postgres:Qwe12345@localhost:5432/FRD')

dfAll =  pd.read_sql_query('select * from "test"',con=engine)


namesOfColumn4Learning.append(nameOfRescanResult)
df =  dfAll[namesOfColumn4Learning]

df80 = df.sample(frac=0.8)
df20 = df.loc[~df.index.isin(df80.index)]

ew = pd.ExcelWriter("C:\\SeScFRD\\Results\\df20Old.xlsx")
df20.to_excel(ew, sheet_name='df20')
ew.save()


NormlizeColumnsValue(df80,df20,namesOfColumn4Normliaze,3)


ew = pd.ExcelWriter("C:\\SeScFRD\\Results\\df20New.xlsx")
df20.to_excel(ew, sheet_name='df20')
ew.save()
    
len = df.shape[1]

# Print
print("dfAll")
print(dfAll.shape)

print("df")
print(df.shape)

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


#Predict

columnsOnlyX = df20.columns[1:len-1]
dfOnlyX = df20[columnsOnlyX]


df20["Predict"] = knn.predict(dfOnlyX)



predictlDf = df20[df20["Predict"] == 1]
rescanResultDf = df20[df20[nameOfRescanResult] == 1]

ew = pd.ExcelWriter("C:\\SeScFRD\\Results\\ResultNew.xlsx")
predictlDf.to_excel(ew, sheet_name='Predict')
rescanResultDf.to_excel(ew, sheet_name='RescanResult')
ew.save() 

print("----------------------")


