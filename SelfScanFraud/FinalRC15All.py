
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

times = 5
testSize = 0.2

def runAlgo( inputPathRescan,inputPathAll, excelWriter, outputFileName ):
    dfRescan = pd.read_excel(inputPathRescan)
    numberOfColumnsWithoutY = dfRescan.shape[1] - 1
    print("dfRescan ",dfRescan.shape)
    print("dfRescan columns ",dfRescan.columns)

    dfResult_1 = dfRescan[dfRescan["RescanResult"] == 1]
    dfResult_0 = dfRescan[dfRescan["RescanResult"] == 0]

    dfAll = pd.read_excel(inputPathAll)
    print("dfAll ",dfAll.shape)
    print("dfAll columns ",dfAll.columns)

    for num in range(0,times):
        print('--   ' + str(num) + '    --')
        dfForBuildModel_1, dfFinalTest_1 = train_test_split(dfResult_1, test_size=testSize, random_state = num*2)
        dfForBuildModel_0, dfFinalTest_0 = train_test_split(dfResult_0, test_size=testSize, random_state = num*2)

        dfForBuildModel = dfForBuildModel_1.append(dfForBuildModel_0)
        dfFinalTest = dfFinalTest_1.append(dfFinalTest_0)
 
        print("dfForBuildModel ",dfForBuildModel.shape)
        print("dfFinalTest ",dfFinalTest.shape)
    
        array1 = dfForBuildModel.values
        dfForBuildModel_x = array1[:,0:numberOfColumnsWithoutY]
        dfForBuildModel_y = array1[:,numberOfColumnsWithoutY]
       
        #dfFinalTestOnlyX
        columnsOnlyX = dfFinalTest.columns[0:numberOfColumnsWithoutY]
        dfFinalTestOnlyX = dfFinalTest[columnsOnlyX]

        dfAllPred = dfFinalTestOnlyX
        for i in range(0,4):
            dfAllPred = dfAllPred.append(dfFinalTestOnlyX)
        dfAllPred = dfAllPred.append(dfAll)

        print("dfFinalTestOnlyX ",dfFinalTestOnlyX.shape)
        print("dfAllPred ",dfAllPred.shape)

        #parameters
        lenOfAllPred = dfAllPred.shape[0]    
        lenOf15PercentAllPred = int(0.15*lenOfAllPred)

        lenOfFinalTest = dfFinalTest.shape[0]    
        lenOf15Percent = int(0.15*lenOfFinalTest)
        lenOfRC15_toFind = int(lenOf15Percent*0.15)

        print("lenOfRC15_toFind ",lenOfRC15_toFind)

        #Find index 15 MLP in dfAllPred
        mlp = MLPClassifier(early_stopping=True,validation_fraction=0.2,max_iter=500)
        mlp.fit(dfForBuildModel_x,dfForBuildModel_y.astype(int))
        predictProbaMLP = mlp.predict_proba(dfAllPred)
        sortedProbMLP = np.sort(predictProbaMLP[:,0])
        prob15AllPredMLP = sortedProbMLP[lenOf15PercentAllPred]

        #Fit MLP
        predictProbaMLP = mlp.predict_proba(dfFinalTestOnlyX)
        dfFinalTest["PredictProbMLP"] = predictProbaMLP[:,0]
        dfFinalTestSort = dfFinalTest.sort_values(by='PredictProbMLP')
        plotPrecsionMLP = GetYPredPrecision(dfFinalTestSort, prob15AllPredMLP,'PredictProbMLP')
        linesForRC15MLP = int(GetRC15_numberOfLines(dfFinalTestSort, lenOfRC15_toFind))
        percentageRC15MLP = linesForRC15MLP/lenOfFinalTest*100
        #probRC15MLP = dfFinalTestSort["PredictProbMLP"].head(linesForRC15MLP).sum() - dfFinalTestSort["PredictProbMLP"].head(linesForRC15MLP -1).sum()
        probRC15MLP = dfFinalTestSort.iloc[linesForRC15MLP]["PredictProbMLP"]
        linesForRC15MLP_AllPred = GetYPredPrecisionAllPred(sortedProbMLP,probRC15MLP)

         #Find index 15 LR in dfAllPred
        lr = LogisticRegression()
        lr.fit(dfForBuildModel_x,dfForBuildModel_y.astype(int))
        predictProbaLR = lr.predict_proba(dfAllPred)
        sortedProbLR = np.sort(predictProbaLR[:,0])
        prob15AllPredLR = sortedProbLR[lenOf15PercentAllPred]

        #Fit LR
        predictProbaLR = lr.predict_proba(dfFinalTestOnlyX)
        dfFinalTest["PredictProabLR"] = predictProbaLR[:,0]
        dfFinalTestSort = dfFinalTest.sort_values(by='PredictProabLR')
        plotPrecsionLR = GetYPredPrecision(dfFinalTestSort, prob15AllPredLR,'PredictProabLR')
        linesForRC15LR = int(GetRC15_numberOfLines(dfFinalTestSort, lenOfRC15_toFind))
        percentageRC15LR = linesForRC15LR/lenOfFinalTest*100
        #probRC15LR = dfFinalTestSort["PredictProabLR"].head(linesForRC15LR).sum() - dfFinalTestSort["PredictProabLR"].head(linesForRC15LR -1).sum()
        probRC15LR = dfFinalTestSort.iloc[linesForRC15LR]["PredictProabLR"]
        linesForRC15LR_AllPred = GetYPredPrecisionAllPred(sortedProbLR,probRC15LR)
        
        ############
        #path = 'C:\\SeScFRD\\Results\\dfFinalTestSortLR_' + outputFileName
        #dfFinalTestSort.to_excel(path, sheet_name='Predict')
        ################

        #Add data to finalResultsDf
        row = [prob15AllPredMLP,plotPrecsionMLP,linesForRC15MLP,percentageRC15MLP,probRC15MLP,linesForRC15MLP_AllPred,\
                prob15AllPredLR,plotPrecsionLR ,linesForRC15LR,percentageRC15LR,probRC15LR,linesForRC15LR_AllPred]
        finalResultsDf.loc[num] = row

    #MLP  
    prob15AllPredMLP = finalResultsDf['prob15AllPredMLP'].mean()
    plotPrecsionMLP = finalResultsDf['plotPrecsionMLP'].mean()
    linesForRC15MLP = finalResultsDf['linesForRC15MLP'].mean()
    percentageRC15MLP = finalResultsDf['percentageRC15MLP'].mean()
    probRC15MLP = finalResultsDf['probRC15MLP'].mean()
    linesForRC15MLP_AllPred = finalResultsDf['linesForRC15AllPredMLP'].mean()

    #LR 
    prob15AllPredLR = finalResultsDf['prob15AllPredLR'].mean()
    plotPrecsionLR = finalResultsDf['plotPrecsionLR'].mean()   
    linesForRC15LR = finalResultsDf['linesForRC15LR'].mean()
    percentageRC15LR = finalResultsDf['percentageRC15LR'].mean()
    probRC15LR = finalResultsDf['probRC15LR'].mean()
    linesForRC15LR_AllPred = finalResultsDf['linesForRC15AllPredLR'].mean()

    #Add data to finalResultsDf Avg
    row = [prob15AllPredMLP,plotPrecsionMLP,linesForRC15MLP,percentageRC15MLP,probRC15MLP,linesForRC15MLP_AllPred,\
                prob15AllPredLR,plotPrecsionLR ,linesForRC15LR,percentageRC15LR,probRC15LR,linesForRC15LR_AllPred]
    finalResultsAvgDf.loc[0] = row

    #write to file
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

def GetYPredPrecision(dfFinalTestSort, probValueToFind, colunmName):
   len = dfFinalTestSort.shape[0]
   for index in range(0,len):
      #probValue = dfFinalTestSort["PredictProbMLP"][index]
      probValue = dfFinalTestSort.iloc[index][colunmName]
      if (probValue >= probValueToFind):
          plot = (dfFinalTestSort["RescanResult"].head(index).sum())/index
          return plot
   return len;

def GetYPredPrecisionAllPred(sortedProbMLP,probValueToFind):
   len = sortedProbMLP.size
   for index in range(0,len):
      probValue = sortedProbMLP[index]
      if (probValue >= probValueToFind):
          return index/len
   return 1;


# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

colomnsNames = ['prob15AllPredMLP','plotPrecsionMLP','linesForRC15MLP','percentageRC15MLP','probRC15MLP','linesForRC15AllPredMLP',\
                'prob15AllPredLR','plotPrecsionLR' ,'linesForRC15LR','percentageRC15LR','probRC15LR','linesForRC15AllPredLR']

finalResultsDf = pd.DataFrame(columns=colomnsNames)
finalResultsAvgDf = pd.DataFrame(columns=colomnsNames)

arr = os.listdir('C:\\SeScFRD\\FinalRescan\\')
for inputFileName in arr:
    outputFileName = inputFileName[:-5] + "_Result.xlsx"
    inputPathAllName =  inputFileName[:-11] + "All.xlsx"
    inputPathRescan  = 'C:\\SeScFRD\\FinalRescan\\' + inputFileName
    outputPath = 'C:\\SeScFRD\\ResultsAllPred\\' + outputFileName
    inputPathAll = 'C:\SeScFRD\FinalAll\\' + inputPathAllName
    ew = pd.ExcelWriter(outputPath)
    runAlgo(inputPathRescan ,inputPathAll , ew, outputFileName)
    ew.save()




