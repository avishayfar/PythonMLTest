
import pandas as pd
import numpy as np
from sqlalchemy import *
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from Factory import GetMLModel
from sklearn.model_selection import train_test_split


def NormlizeColumnsValue (df80, df20, namesOfColumns, threshold):
    for columnName in namesOfColumns:
        basicStd  = df80[columnName].std()
        basicMean = df80[columnName].mean()
        NormlizeColumnValue (df80, columnName, basicMean, basicStd, threshold)
        NormlizeColumnValue (df20, columnName, basicMean, basicStd, threshold)


def NormlizeColumnValue (df, columnName, basicMean, basicStd, threshold):
     maxValue = threshold
     minValue = (-1)*threshold
     std4Calc = 1 if basicStd == 0 else basicStd
     df[columnName] = df[columnName].pipe(lambda x: (x-basicMean)/std4Calc)
     df[columnName][df[columnName] > maxValue] = maxValue
     df[columnName][df[columnName] < minValue] = minValue
       

 ########################################################################

def RunFraudDetector( algorithmName, namesOfColumn4Learning, namesOfColumn4Normliaze, rescanResultColumnName, shrinkageColumnName, threshold, OriginRescanRate):
   
    tempPath = "C:\\SeScFRD\\Input\\dataWithHierarchies.xlsx"
    dfAll = pd.read_excel(tempPath)

    namesOfAllColumns = namesOfColumn4Learning.copy()
    namesOfAllColumns.append(rescanResultColumnName)
    namesOfAllColumns.append(shrinkageColumnName)

    dfAll["PosTransactionId"] = dfAll["PosTransactionId"].str[9:-12]
    dfAll["PosTransactionId"] = pd.to_datetime(dfAll["PosTransactionId"], format='%m/%d/%Y')
    dfAll = dfAll.sort_values(by="PosTransactionId",ascending = True)

    df = dfAll[namesOfAllColumns]
   
    OriginalFraudPrecision = dfResult_1.shape[0]/df.shape[0]

    shrinkageSum = df[shrinkageColumnName].sum()
    originalShrinkagePerFruadLine = shrinkageSum/(df.shape[0])

    ResultsDf = pd.DataFrame(columns=['Precision Of Data', 'Model' , 'Pivot Value', 'Precsion Of Predict Fruad', 'Shrinkage Per Line'])
    for num in range(1,99):

        #Build dfForBuildModel dfFinalTest 
        BuildModelSize = int(df.shape[0] * num/100)
        testSize = 100
        dfForBuildModel = df.head(BuildModelSize)
        dfFinalTest = df[BuildModelSize+1, BuildModelSize+101]

        if(namesOfColumn4Normliaze != ""):
            NormlizeColumnsValue(dfForBuildModel, dfFinalTest,namesOfColumn4Normliaze,3)
    
        #print data
        print()
        print(".....................................")
        print("All data size - ", df.shape)
        print("The precsion of fraud - ", OriginalFraudPrecision)
        print("dfForBuildModel size - ", dfForBuildModel.shape)
        print("dfFinalTest size - ", dfFinalTest.shape)

        # Build data for the machine learning 

        dfForBuildModel_x = dfForBuildModel[namesOfColumn4Learning]
        dfForBuildModel_y = dfForBuildModel[rescanResultColumnName]
           
        #Start machine learning.

        model = GetMLModel(algorithmName)
        model.fit(dfForBuildModel_x, dfForBuildModel_y.astype(int))

        print("model -", algorithmName)

        #Predict

        dfOnlyX = dfFinalTest[namesOfColumn4Learning]
        dfFinalTest["Predict"] = model.predict_proba(dfOnlyX)[:,0]  

        #Get specific data information

        numberOfAllRowsFinalTest = dfFinalTest.shape[0] 
        numberOfRows4Anlayze = int(numberOfAllRowsFinalTest * OriginRescanRate/100)

        dfFinalTest = dfFinalTest.sort_values(by="Predict",ascending = True)
        df4Analayze = dfFinalTest.head(numberOfRows4Anlayze)
        pivotValue = dfFinalTest.iloc[numberOfRows4Anlayze]["Predict"]
        numberOfTruePredict = (df4Analayze[rescanResultColumnName][df4Analayze[rescanResultColumnName] > 0]).sum()
        precsionOfShrinkage4Line = df4Analayze[shrinkageColumnName].sum()/numberOfRows4Anlayze 
        precisionOfTruePredict = numberOfTruePredict/df4Analayze.shape[0]*100

        numberOfRealFraudLinesToBeCatch = round(OriginalFraudPrecision * numberOfRows4Anlayze) 
        precisionOfRecall4GetPreDefinedFraud = GetRecallPrecision(dfFinalTest, rescanResultColumnName, numberOfRealFraudLinesToBeCatch)

        print("numberOfAllRowsFinalTest - ", numberOfAllRowsFinalTest)
        print("numberOfRows4Anlayze - ", numberOfRows4Anlayze)
        print("precisionOfTruePredict - ", precisionOfTruePredict) 
        print("precsionOfShrinkage4Line - ", precsionOfShrinkage4Line)
        print("precisionOfRecall4GetPreDefinedFraud - ", precisionOfRecall4GetPreDefinedFraud)
        print(".......................................")
        print()

        ResultsDf.loc[num-1] = [num,algorithmName, pivotValue, precisionOfTruePredict, precsionOfShrinkage4Line]

    return ResultsDf

def GetRecallPrecision(dfWithScanResult, rescanResultColumnName, numberOfRealFraudLinesToBeCatch):
   len = dfWithScanResult.shape[0]
  
   numberOfResult = 0
   for num in range(0,len):
      numberOfResult += dfWithScanResult[rescanResultColumnName].iloc[num]
      if (numberOfRealFraudLinesToBeCatch == numberOfResult):
          recallPrcesion = num/len*100
          return recallPrcesion;
   return 100;

