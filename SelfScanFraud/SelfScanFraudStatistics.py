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
from sklearn.model_selection import train_test_split

times = 1000
testSize = 0.2

def runAlgo( model, name, inputPath, excelWriter ):
    df = pd.read_excel(inputPath)
    len = df.shape[1]
    for num in range(0,times):
        dfForBuildModel, dfFinalTest = train_test_split(df, test_size=testSize)

        print("df ",df.shape)
        print("dfForBuildModel ",dfForBuildModel.shape)
        print("dfFinalTest ",dfFinalTest.shape)
    
        array1 = dfForBuildModel.values
        data_x = array1[:,0:len -1]
        data_y = array1[:,len - 1]
        x_train, x_test, y_train, y_test = model_selection.train_test_split (data_x, data_y, test_size = testSize, random_state = 42)

        columnsOnlyX = dfFinalTest.columns[0:len-1]
        dfOnlyX = dfFinalTest[columnsOnlyX]

        #parameters
        lenOfFinalTest = dfFinalTest.shape[0] 
        lenOf10Percent = int(lenOfFinalTest/10)
        lenOf15Percent = int(0.15*lenOfFinalTest)
        lenOf20Percent = int(lenOfFinalTest/5) 

        #Fit
        model.fit(x_train,y_train.astype(int))
        predictProba = model.predict_proba(dfOnlyX)

        #Sort
        dfFinalTest["PredictProb"] = predictProba[:,0]
        dfFinalTestSort = dfFinalTest.sort_values(by='PredictProb')

        avg = dfFinalTestSort["RescanResult"].mean()
        avg10 = dfFinalTestSort["RescanResult"].head(lenOf10Percent).mean()
        avg15 = dfFinalTestSort["RescanResult"].head(lenOf15Percent).mean()
        avg20 = dfFinalTestSort["RescanResult"].head(lenOf20Percent).mean()
        improve10 = avg10/avg*100
        improve15 = avg15/avg*100
        improve20 = avg20/avg*100
      

        print("Avg ", avg)
        print("--------")

        #Add data to finalResultsDf
        row = [avg,avg10,avg15,avg20,improve10,improve20]
        finalResultsDf.loc[num] = row

    avg = finalResultsDf['Avg'].mean()
    avg10 = finalResultsDf['Avg10'].mean()
    avg10 = finalResultsDf['Avg15'].mean()
    avg20 = finalResultsDf['Avg20'].mean()
    improve10 = finalResultsDf['Improve10'].mean()
    improve20 = finalResultsDf['Improve20'].mean()

    row = [avg,avg10,avg15,avg20,improve10,improve20]
    finalResultsAvgDf.loc[0] = row

    sheetName = name + ' Final All'
    finalResultsDf.to_excel(excelWriter, sheet_name=sheetName)
    sheetName = name + ' Final Avg All'
    finalResultsAvgDf.to_excel(excelWriter, sheet_name=sheetName);
   

# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

colomnsNames = ['Avg','Avg10','Avg15','Avg20','Improve10','Improve20']
finalResultsDf = pd.DataFrame(columns=colomnsNames)
finalResultsAvgDf = pd.DataFrame(columns=colomnsNames)

filesNames = []
#filesNames.append(('OnlyRescanData.xlsx', 'SelfScanResults.xlsx'))
filesNames.append(('phase1.xlsx', 'phase1_Results.xlsx'))
filesNames.append(('phase2.xlsx', 'phase2_Results.xlsx'))
filesNames.append(('phase3.xlsx', 'phase3_Results.xlsx'))
filesNames.append(('phase4.xlsx', 'phase4_Results.xlsx'))
filesNames.append(('phase5.xlsx', 'phase5_Results.xlsx'))
filesNames.append(('phase6.xlsx', 'phase6_Results.xlsx'))
filesNames.append(('phase7.xlsx', 'phase7_Results.xlsx'))
filesNames.append(('phase5Truncated.xlsx', 'phase5Truncated_Results.xlsx'))

models = []
models.append((MLPClassifier(), 'MLP_C'))
#models.append((SVC(probability=True), 'SVC'))
#models.append((DecisionTreeClassifier(), 'DTC'))
models.append((LogisticRegression(), 'LR'))
#models.append((KNeighborsClassifier(), 'KNN'))

for inputFileName, outputFileName in filesNames:
    inputPath  = 'C:\\SeScFRD\\' + inputFileName
    outputPath = 'C:\\SeScFRD\\Results\\' + outputFileName
    ew = pd.ExcelWriter(outputPath)
    for model, name in models:
        runAlgo( model , name , inputPath , ew)
    ew.save()

