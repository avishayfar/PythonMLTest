import pandas as pd
import numpy as np
import os
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

times = 1
testSize = 0.2


def runAlgo( inputPath, excelWriter, outputFileName ):
    df = pd.read_excel(inputPath)
    numberOfColumns = df.shape[1]

    dfResult_1 = df[df["RescanResult"] == 1]
    dfResult_0 = df[df["RescanResult"] == 0]

    for num in range(0,times):
        dfForBuildModel_1, dfFinalTest_1 = train_test_split(dfResult_1, test_size=testSize, random_state = 51)
        dfForBuildModel_0, dfFinalTest_0 = train_test_split(dfResult_0, test_size=testSize, random_state = 51)

        dfForBuildModel = dfForBuildModel_1.append(dfForBuildModel_0)
        dfFinalTest = dfFinalTest_1.append(dfFinalTest_0)

        print("df ",df.shape)
        print("numberOfColumns ",numberOfColumns)
        print("dfForBuildModel ",dfForBuildModel.shape)
        print("dfFinalTest ",dfFinalTest.shape)
    
        array1 = dfForBuildModel.values
        data_x = array1[:,0:numberOfColumns -1]
        data_y = array1[:,numberOfColumns - 1]
        x_train, x_test, y_train, y_test = model_selection.train_test_split (data_x, data_y, test_size = testSize, random_state = 42)

        columnsOnlyX = dfFinalTest.columns[0:numberOfColumns-1]
        dfFinalTestOnlyX = dfFinalTest[columnsOnlyX]

        #parameters
        lenOfFinalTest = dfFinalTest.shape[0] 
        lenOf10Percent = int(lenOfFinalTest/10)
        lenOf15Percent = int(0.15*lenOfFinalTest)
        lenOf20Percent = int(lenOfFinalTest/5)
        lenOfRC15_toFind = int(lenOf15Percent*0.15)


        print("lenOfRC15_toFind ",lenOfRC15_toFind)

        #Fit MLP
        mlp = MLPClassifier()
        mlp.fit(x_train,y_train.astype(int))
        predictProbaMLP = mlp.predict_proba(dfFinalTestOnlyX)
        dfFinalTest["PredictProbMLP"] = predictProbaMLP[:,0]

        #Fit LR
        lr = LogisticRegression()
        lr.fit(x_train,y_train.astype(int))
        predictProbaLR = lr.predict_proba(dfFinalTestOnlyX)
        dfFinalTest["PredictProabLR"] = predictProbaLR[:,0]

        #avg     
        avg = dfFinalTest["RescanResult"].mean()
      
        #MLP Sort     
        dfFinalTestSort = dfFinalTest.sort_values(by='PredictProbMLP')
        avg10MLP = dfFinalTestSort["RescanResult"].head(lenOf10Percent).mean()
        avg15MLP = dfFinalTestSort["RescanResult"].head(lenOf15Percent).mean()
        avg20MLP = dfFinalTestSort["RescanResult"].head(lenOf20Percent).mean()
        improve10MLP = avg10MLP/avg*100
        improve15MLP = avg15MLP/avg*100
        improve20MLP = avg20MLP/avg*100
        linesForRC15MLP = GetRC15_numberOfLines(dfFinalTestSort, lenOfRC15_toFind)

        #LR Sort
        dfFinalTestSort = dfFinalTest.sort_values(by='PredictProabLR')
        avg10LR = dfFinalTestSort["RescanResult"].head(lenOf10Percent).mean()
        avg15LR = dfFinalTestSort["RescanResult"].head(lenOf15Percent).mean()
        avg20LR = dfFinalTestSort["RescanResult"].head(lenOf20Percent).mean()
        improve10LR = avg10LR/avg*100
        improve15LR = avg15LR/avg*100
        improve20LR = avg20LR/avg*100
        linesForRC15LR = GetRC15_numberOfLines(dfFinalTestSort, lenOfRC15_toFind)
        
        ############
        path = 'C:\\SeScFRD\\Results\\dfFinalTestSortLR_' + outputFileName
        dfFinalTestSort.to_excel(path, sheet_name='Predict')
        ################

        #Add data to finalResultsDf
        row = [avg, avg10MLP ,avg15MLP ,avg20MLP ,improve10MLP, improve15MLP ,improve20MLP, linesForRC15MLP, avg10LR ,avg15LR ,avg20LR ,improve10LR, improve15LR ,improve20LR,linesForRC15LR]
        finalResultsDf.loc[num] = row

    avg = finalResultsDf['Avg'].mean()
    #MLP  
    avg10MLP = finalResultsDf['Avg10_MLP'].mean()
    avg15MLP = finalResultsDf['Avg15_MLP'].mean()
    avg20MLP = finalResultsDf['Avg20_MLP'].mean()
    improve10MLP = finalResultsDf['Improve10_MLP'].mean()
    improve15MLP = finalResultsDf['Improve15_MLP'].mean()
    improve20MLP = finalResultsDf['Improve20_MLP'].mean()
    linesForRC15MLP = finalResultsDf['linesForRC15MLP'].mean()
    #LR
    avg10LR = finalResultsDf['Avg10_LR'].mean()
    avg15LR = finalResultsDf['Avg15_LR'].mean()
    avg20LR = finalResultsDf['Avg20_LR'].mean()
    improve10LR = finalResultsDf['Improve10_LR'].mean()
    improve15LR = finalResultsDf['Improve15_LR'].mean()
    improve20LR = finalResultsDf['Improve20_LR'].mean()
    linesForRC15LR = finalResultsDf['linesForRC15LR'].mean()

    row = [avg, avg10MLP ,avg15MLP ,avg20MLP ,improve10MLP, improve15MLP ,improve20MLP,linesForRC15MLP, avg10LR ,avg15LR ,avg20LR ,improve10LR, improve15LR ,improve20LR, linesForRC15LR]
    finalResultsAvgDf.loc[0] = row

    sheetName = ' Final All'
    finalResultsDf.to_excel(excelWriter, sheet_name=sheetName)
    sheetName = ' Final Avg All'
    finalResultsAvgDf.to_excel(excelWriter, sheet_name=sheetName);

def GetRC15_numberOfLines(dfWithScanResult, lenOfRC15_toFind):
   len = dfWithScanResult.shape[0]
   for num in range(0,len):
      numberOfResult1 = dfWithScanResult["RescanResult"].head(num).sum()
      if (lenOfRC15_toFind == numberOfResult1):
          return num
   return len;

# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

colomnsNames = ['Avg','Avg10_MLP','Avg15_MLP','Avg20_MLP','Improve10_MLP', 'Improve15_MLP','Improve20_MLP', 'linesForRC15MLP','Avg10_LR','Avg15_LR','Avg20_LR','Improve10_LR','Improve15_LR','Improve20_LR','linesForRC15LR']
finalResultsDf = pd.DataFrame(columns=colomnsNames)
finalResultsAvgDf = pd.DataFrame(columns=colomnsNames)

arr = os.listdir('C:\\SeScFRD\\New\\')
for inputFileName in arr:
    outputFileName = inputFileName[:-5] + "_Result.xlsx"
    inputPath  = 'C:\\SeScFRD\\New\\' + inputFileName
    outputPath = 'C:\\SeScFRD\\Results\\' + outputFileName
    ew = pd.ExcelWriter(outputPath)
    runAlgo(inputPath , ew, outputFileName)
    ew.save()




