import pandas as pd
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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

times = 15
percentage15Size = 48

def runAlgo( model, name, excelWriter ):
    for num in range(0,times):
        df80 = df.sample(frac=0.8)
        df20 = df.loc[~df.index.isin(df80.index)]
    
        array = df80.values
        data_x = array[:,0:len -1]
        data_y = array[:,len - 1]
        x_train, x_test, y_train, y_test   =  model_selection.train_test_split (data_x, data_y, test_size = 0.2, random_state = 42)

        columnsOnlyX = df20.columns[0:len-1]
        dfOnlyX = df20[columnsOnlyX]

        #Model
        model.fit(x_train,y_train.astype(int))
        predictProba = model.predict_proba(dfOnlyX)

        df20["PredictProb"] = predictProba[:,0]
        df20Sort = df20.sort_values(by='PredictProb')
        avg = df20Sort["RescanResult"].mean()
        avg15 = df20Sort["RescanResult"].head(percentage15Size).mean()
        print("Avg ", avg)
        print("Avg15 ", avg15)
        improve = avg15/avg*100
        print("Improve ", improve)
        print("--------")

        row = [avg,avg15,improve]
        finalResultsDf.loc[num] = row

    avg = finalResultsDf['Avg'].mean()
    avg15 = finalResultsDf['Avg15'].mean()
    improve = finalResultsDf['Improve'].mean()
    row = [avg,avg15,improve]
    finalResultsAvgDf.loc[0] = row

    sheetName = name + ' Final All'
    finalResultsDf.to_excel(excelWriter, sheet_name=sheetName)
    sheetName = name + ' Final Avg All'
    finalResultsAvgDf.to_excel(excelWriter, sheet_name=sheetName);
   

# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

url = "C:\\SeScFRD\\OnlyRescanData.xlsx" 

df = pd.read_excel(url)

df80 = df.sample(frac=0.8)
df20 = df.loc[~df.index.isin(df80.index)]

len = df.shape[1]

# shape
print("SeSc df")
print(df.shape)

print("SeSc df80")
print(df80.shape)

print("SeSc df20")
print(df20.shape)

colomnsNames = ['Avg','Avg15','Improve']
finalResultsDf = pd.DataFrame(columns=colomnsNames)
finalResultsAvgDf = pd.DataFrame(columns=colomnsNames)

path = 'C:\\SeScFRD\\SelfScanResults.xlsx'
ew = pd.ExcelWriter(path)

models = []
models.append((MLPClassifier(), 'MLP_C'))
models.append((SVC(probability=True), 'SVC'))
models.append((DecisionTreeClassifier(), 'DTC'))
models.append((LogisticRegression(), 'LR'))
models.append((KNeighborsClassifier(), 'KNN'))

for model, name in models:
    runAlgo( model , name , ew)

ew.save()

