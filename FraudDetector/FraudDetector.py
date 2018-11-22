import pandas as pd
import numpy as np
from sqlalchemy import *
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from Factory import GetMLModel
from sklearn.model_selection import train_test_split


testSize = 0.2


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

def RunFraudDetector(basicDataTableName, byPrecision, algorithmName, namesOfColumn4Learning, namesOfColumn4Normliaze, rescanResultColumnName, shrinkageColumnName, times, threshold, OriginRescanRate):

    #engine = create_engine('postgresql://postgres:Qwe12345@localhost:5432/FRD')
    #query = 'select * from "{}"'.format(basicDataTableName)   
    #dfAll =  pd.read_sql_query(query, con=engine)

    #tempPath = "C:\\SeScFRD\\Input\\ItamarData.xlsx"
    tempPath = "C:\\SeScFRD\\Input\\dataWithHierarchies.xlsx"
    dfAll = pd.read_excel(tempPath)


    namesOfAllColumns = namesOfColumn4Learning.copy()
    namesOfAllColumns.append(rescanResultColumnName)
    namesOfAllColumns.append(shrinkageColumnName)

    if(byPrecision>0):
        dfAll["PosTransactionId"] = dfAll["PosTransactionId"].str[9:-12]
        dfAll["PosTransactionId"] = pd.to_datetime(dfAll["PosTransactionId"], format='%m/%d/%Y')
        dfAll = dfAll.sort_values(by="PosTransactionId",ascending = True)

    df = dfAll[namesOfAllColumns]

    dfResult_1 = df[df[rescanResultColumnName] == 1]
    dfResult_0 = df[df[rescanResultColumnName] == 0]

    OriginalFraudPrecision = dfResult_1.shape[0]/df.shape[0]

    shrinkageSum = df[shrinkageColumnName].sum()
    originalShrinkagePerFruadLine = shrinkageSum/(df.shape[0])

    if(byPrecision>1):
        times = 1
    numberOfTruePredictList = []
    precisionOfLines4RecallList = []
    precsionOfShrinkage4LineList = []
    for num in range(0,times):

        #Build dfForBuildModel dfFinalTest 
        if(byPrecision==0):
            dfForBuildModel_1, dfFinalTest_1 = train_test_split(dfResult_1, test_size=testSize)
            dfForBuildModel_0, dfFinalTest_0 = train_test_split(dfResult_0, test_size=testSize)

            dfForBuildModel = dfForBuildModel_1.append(dfForBuildModel_0)
            dfFinalTest = dfFinalTest_1.append(dfFinalTest_0)
        else:
            head = int(df.shape[0] * byPrecision/100)
            tail = df.shape[0] - head
            dfForBuildModel = df.head(head)
            dfFinalTest = df.tail(tail)

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
        print("pivotValue - ", pivotValue)
        print("precisionOfTruePredict - ", precisionOfTruePredict) 
        print("precsionOfShrinkage4Line - ", precsionOfShrinkage4Line)
        print("precisionOfRecall4GetPreDefinedFraud - ", precisionOfRecall4GetPreDefinedFraud)
        print(".......................................")
        print()

        numberOfTruePredictList.append(precisionOfTruePredict)
        precisionOfLines4RecallList.append(precisionOfRecall4GetPreDefinedFraud)
        precsionOfShrinkage4LineList.append(precsionOfShrinkage4Line)

    numberOfTruePredictNdList = np.array(numberOfTruePredictList)
    precisionOfRecalLineslNdList = np.array(precisionOfLines4RecallList)
    precsionOfShrinkage4LineNdList = np.array(precsionOfShrinkage4LineList)

    FraudResults = {}
    FraudResults["Model"] = algorithmName
    FraudResults["Times"] = times
    #Precsion Of true predict
    FraudResults["PrecsionOfOriginalFraud"] = OriginalFraudPrecision*100
    FraudResults["PrecsionOfTruePredict"] = numberOfTruePredictNdList.mean()
    #Recall
    FraudResults["OriginalPrecisionOfLinesForRecall"] = OriginRescanRate
    FraudResults["PrecisionOfLinesForRecall"] = precisionOfRecalLineslNdList.mean()
    #Shrinkage
    FraudResults["OriginalShrinkagePerLine"] = originalShrinkagePerFruadLine
    FraudResults["ShrinkagePerLine"] = precsionOfShrinkage4LineNdList.mean()

    return FraudResults

def GetRecallPrecision(dfWithScanResult, rescanResultColumnName, numberOfRealFraudLinesToBeCatch):
   len = dfWithScanResult.shape[0]
  
   numberOfResult = 0
   for num in range(0,len):
      numberOfResult += dfWithScanResult[rescanResultColumnName].iloc[num]
      if (numberOfRealFraudLinesToBeCatch == numberOfResult):
          recallPrcesion = num/len*100
          return recallPrcesion;
   return 100;

