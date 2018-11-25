
import pandas as pd
import numpy as np
from sqlalchemy import *
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from Factory import GetMLModel
from sklearn.model_selection import train_test_split


def NormlizeColumnsValue (df1, df2, namesOfColumns, threshold):
    for columnName in namesOfColumns:
        basicStd  = df1[columnName].std()
        basicMean = df1[columnName].mean()
        NormlizeColumnValue (df1, columnName, basicMean, basicStd, threshold)
        NormlizeColumnValue (df2, columnName, basicMean, basicStd, threshold)


def NormlizeColumnValue (df, columnName, basicMean, basicStd, threshold):
     maxValue = threshold
     minValue = (-1)*threshold
     std4Calc = 1 if basicStd == 0 else basicStd
     df[columnName] = df[columnName].pipe(lambda x: (x-basicMean)/std4Calc)
     df[columnName][df[columnName] > maxValue] = maxValue
     df[columnName][df[columnName] < minValue] = minValue
       

 ########################################################################

def RunFraudDetectorSelfLearning( algorithmName, namesOfColumn4Learning, namesOfColumn4Normliaze, rescanResultColumnName, shrinkageColumnName, threshold, OriginRescanRate):
   
    tempPath = "C:\\SeScFRD\\Input\\dataWithHierarchies.xlsx"
    dfAll = pd.read_excel(tempPath)

    namesOfAllColumns = namesOfColumn4Learning.copy()
    namesOfAllColumns.append(rescanResultColumnName)
    namesOfAllColumns.append(shrinkageColumnName)

    dfAll["PosTransactionId"] = dfAll["PosTransactionId"].str[9:-12]
    dfAll["PosTransactionId"] = pd.to_datetime(dfAll["PosTransactionId"], format='%m/%d/%Y')
    dfAll = dfAll.sort_values(by="PosTransactionId",ascending = True)

    df = dfAll[namesOfAllColumns]
    dfLength = df.shape[0]
    
    dfForBuildModel = df[0 : int(dfLength*0.45)]
    df4Analayze = pd.DataFrame();

    pivotValue = 0.3

    shrinkageSum = df[shrinkageColumnName].sum()
    originalShrinkagePerFruadLine = shrinkageSum/(df.shape[0])

    ResultsDf = pd.DataFrame(columns=['Precision Of Data', 'Model' , 'Pivot Value', 'Precision Of Analayze', 'Precsion Of Predict Fruad', 'Shrinkage Per Line'])
    for num in range(45,86,10):

        #Build dfForBuildModel dfFinalTest 
        dfForBuildModel = dfForBuildModel.append(df4Analayze)
        
        testSize = int(df.shape[0] /10)
        dfFinalTestStartIndex = int(df.shape[0] * num/100)

        #dfForBuildModel = df[0: dfFinalTestStartIndex -1] 
        dfFinalTest = df[dfFinalTestStartIndex : dfFinalTestStartIndex + testSize]

        if(namesOfColumn4Normliaze != ""):
            NormlizeColumnsValue(dfForBuildModel, dfFinalTest, namesOfColumn4Normliaze,3)
    
        #print data
        print()
        print(".....................................")
        print("add to dfForBuildModel - ", df4Analayze.shape)
        print("dfForBuildModel size   - ", dfForBuildModel.shape)
        print("dfFinalTest size       - ", dfFinalTest.shape)
        print("-->")

        # Build data for the machine learning 

        dfForBuildModel_x = dfForBuildModel[namesOfColumn4Learning]
        dfForBuildModel_y = dfForBuildModel[rescanResultColumnName]
           
        #Start machine learning.

        model = GetMLModel(algorithmName)
        model.fit(dfForBuildModel_x, dfForBuildModel_y.astype(int))
       
        #Predict dfFinalTest
        dfOnlyX = dfFinalTest[namesOfColumn4Learning]
        dfFinalTest["Predict"] = model.predict_proba(dfOnlyX)[:,1]  

        #Find the true predict
        numberOfAllRowsFinalTest = dfFinalTest.shape[0] 
        numberOfRows4Anlayze = int(numberOfAllRowsFinalTest * OriginRescanRate/100)

        #Find pivot   - temp
        #pivotValue = dfFinalTest["Predict"].sort_values(ascending = False).head(numberOfRows4Anlayze).tail(1).values[0]

        #Analayze
        df4Analayze = dfFinalTest[dfFinalTest["Predict"] > pivotValue]
        numberOfRows4Anlayze = df4Analayze.shape[0]
        numberOfTruePredict = (df4Analayze[rescanResultColumnName][df4Analayze[rescanResultColumnName] > 0]).sum()
        precsionOfShrinkage4Line = df4Analayze[shrinkageColumnName].sum()/numberOfRows4Anlayze 
        precisionOfTruePredict = numberOfTruePredict/numberOfRows4Anlayze*100

      

        print("Precision of data- ", num)
        print("numberOfAllRowsFinalTest - ", dfFinalTest.shape[0])
        print("numberOfRows4Anlayze - ", numberOfRows4Anlayze)
        precisionOfAnalayze = numberOfRows4Anlayze/dfFinalTest.shape[0]*100
        print("precisionOfAnalayze- ", precisionOfAnalayze)
        print("pivotValue - ", pivotValue)
        print("precisionOfTruePredict - ", precisionOfTruePredict) 
        print(".......................................")
        print()

        ResultsDf.loc[num-1] = [num,algorithmName, pivotValue,precisionOfAnalayze, precisionOfTruePredict, precsionOfShrinkage4Line]

    return ResultsDf




#import pandas as pd
#import numpy as np
#from sqlalchemy import *
#from pandas.tools.plotting import scatter_matrix
#import matplotlib.pyplot as plt
#from sklearn import model_selection
#from sklearn.externals import joblib
#from Factory import GetMLModel
#from sklearn.model_selection import train_test_split


#def NormlizeColumnsValue (df1, df2, df3, namesOfColumns, threshold):
#    for columnName in namesOfColumns:
#        basicStd  = df1[columnName].std()
#        basicMean = df1[columnName].mean()
#        NormlizeColumnValue (df1, columnName, basicMean, basicStd, threshold)
#        NormlizeColumnValue (df2, columnName, basicMean, basicStd, threshold)
#        NormlizeColumnValue (df3, columnName, basicMean, basicStd, threshold)


#def NormlizeColumnValue (df, columnName, basicMean, basicStd, threshold):
#     maxValue = threshold
#     minValue = (-1)*threshold
#     std4Calc = 1 if basicStd == 0 else basicStd
#     df[columnName] = df[columnName].pipe(lambda x: (x-basicMean)/std4Calc)
#     df[columnName][df[columnName] > maxValue] = maxValue
#     df[columnName][df[columnName] < minValue] = minValue
       

# ########################################################################

#def RunFraudDetectorSelfLearning( algorithmName, namesOfColumn4Learning, namesOfColumn4Normliaze, rescanResultColumnName, shrinkageColumnName, threshold, OriginRescanRate):
   
#    tempPath = "C:\\SeScFRD\\Input\\dataWithHierarchies.xlsx"
#    dfAll = pd.read_excel(tempPath)

#    namesOfAllColumns = namesOfColumn4Learning.copy()
#    namesOfAllColumns.append(rescanResultColumnName)
#    namesOfAllColumns.append(shrinkageColumnName)

#    dfAll["PosTransactionId"] = dfAll["PosTransactionId"].str[9:-12]
#    dfAll["PosTransactionId"] = pd.to_datetime(dfAll["PosTransactionId"], format='%m/%d/%Y')
#    dfAll = dfAll.sort_values(by="PosTransactionId",ascending = True)

#    df = dfAll[namesOfAllColumns]
#    dfLength = df.shape[0]
#    df4FindPivotSize = int(dfLength/20)

#    df4FindPivot = df[0 : df4FindPivotSize]
#    dfForBuildModel = df[df4FindPivotSize: int(dfLength*0.90)]
#    df4Analayze = pd.DataFrame();


#    shrinkageSum = df[shrinkageColumnName].sum()
#    originalShrinkagePerFruadLine = shrinkageSum/(df.shape[0])

#    ResultsDf = pd.DataFrame(columns=['Precision Of Data', 'Model' , 'Pivot Value', 'Precision Of Analayze', 'Precsion Of Predict Fruad', 'Shrinkage Per Line'])
#    for num in range(45,86,10):

#        #Build dfForBuildModel dfFinalTest 
#        dfForBuildModel = dfForBuildModel.append(df4Analayze)
        
#        testSize = int(df.shape[0] /10)
#        dfFinalTestStartIndex = int(df.shape[0] * num/100)
#        dfFinalTest = df[dfFinalTestStartIndex : dfFinalTestStartIndex + testSize]

#        if(namesOfColumn4Normliaze != ""):
#            NormlizeColumnsValue(dfForBuildModel, dfFinalTest, df4FindPivot, namesOfColumn4Normliaze,3)
    
#        #print data
#        print()
#        print(".....................................")
#        print("add to dfForBuildModel - ", df4Analayze.shape)
#        print("dfForBuildModel size   - ", dfForBuildModel.shape)
#        print("df4FindPivotSize size  - ", df4FindPivot.shape)
#        print("dfFinalTest size       - ", dfFinalTest.shape)
#        print("-->")

#        # Build data for the machine learning 

#        dfForBuildModel_x = dfForBuildModel[namesOfColumn4Learning]
#        dfForBuildModel_y = dfForBuildModel[rescanResultColumnName]
           
#        #Start machine learning.

#        model = GetMLModel(algorithmName)
#        model.fit(dfForBuildModel_x, dfForBuildModel_y.astype(int))

#        #Predict df4FindPivot
#        dfOnlyX = df4FindPivot[namesOfColumn4Learning]
#        df4FindPivot["Predict"] = model.predict_proba(dfOnlyX)[:,1]  

#        #Find the Pivot
#        numberOfAllRowsFinalTest = df4FindPivot.shape[0] 
#        numberOfRows4Anlayze = int(numberOfAllRowsFinalTest * OriginRescanRate/100)

#        df4FindPivot = df4FindPivot.sort_values(by="Predict",ascending = False)
#        df4Analayze = df4FindPivot.head(numberOfRows4Anlayze)
#        pivotValue = df4FindPivot.iloc[numberOfRows4Anlayze]["Predict"]
       
#        #Predict dfFinalTest
#        dfOnlyX = dfFinalTest[namesOfColumn4Learning]
#        dfFinalTest["Predict"] = model.predict_proba(dfOnlyX)[:,1]  

#        #Find the true predict
#        numberOfAllRowsFinalTest = dfFinalTest.shape[0] 
#        numberOfRows4Anlayze = int(numberOfAllRowsFinalTest * OriginRescanRate/100)

#        df4Analayze = dfFinalTest[dfFinalTest["Predict"] > pivotValue]
#        numberOfRows4Anlayze = df4Analayze.shape[0]
#        pivotValueb = df4Analayze.iloc[numberOfRows4Anlayze-1]["Predict"]
#        numberOfTruePredict = (df4Analayze[rescanResultColumnName][df4Analayze[rescanResultColumnName] > 0]).sum()
#        precsionOfShrinkage4Line = df4Analayze[shrinkageColumnName].sum()/numberOfRows4Anlayze 
#        precisionOfTruePredict = numberOfTruePredict/df4Analayze.shape[0]*100

#        #df4Analayze = dfFinalTest[dfFinalTest["Predict"] > pivotValue]

#        print("Precision of data- ", num)
#        print("numberOfAllRowsFinalTest - ", dfFinalTest.shape[0])
#        print("numberOfRows4Anlayze - ", df4Analayze.shape[0])
#        precisionOfAnalayze = df4Analayze.shape[0]/dfFinalTest.shape[0]*100
#        print("precisionOfAnalayze- ", precisionOfAnalayze)
#        print("pivotValue - ", pivotValue)
#        print("pivotValueb - ", pivotValueb)
#        print("precisionOfTruePredict - ", precisionOfTruePredict) 
#        print(".......................................")
#        print()

#        ResultsDf.loc[num-1] = [num,algorithmName, pivotValue,precisionOfAnalayze, precisionOfTruePredict, precsionOfShrinkage4Line]

#    return ResultsDf







