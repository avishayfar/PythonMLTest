import pandas as pd
import numpy as np
from sqlalchemy import *
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from Factory import GetMLModel


def NormlizeColumnsValue (df80, df20, namesOfColumns, threshold):
    for columnName in namesOfColumns:
        basicStd  = df80[columnName].std()
        basicMean = df80[columnName].mean()
        NormlizeColumnValue (df80, columnName, basicMean, basicStd, threshold)
        NormlizeColumnValue (df20, columnName, basicMean, basicStd, threshold)

def NormlizeColumnValue (df, columnName, basicMean, basicStd, threshold):
     maxValue = basicStd * threshold
     minValue = (-1)*maxValue
     df[columnName] = df[columnName].pipe(lambda x: (x-basicMean) / basicStd)
     df[columnName][df[columnName] > maxValue] = maxValue
     df[columnName][df[columnName] < minValue] = minValue
       

 ########################################################################

def RunFraudDetector(basicDataTableName, storeWithHierarchyName, algorithmName, namesOfColumn4Learning, namesOfColumn4Normliaze, nameOfRescanResult, times, threshold, precision):

    engine = create_engine('postgresql://postgres:Qwe12345@localhost:5432/FRD')

    query = 'select * from "{}"'.format(basicDataTableName)
    dfAll =  pd.read_sql_query(query, con=engine)

    namesOfColumn4Learning.append(nameOfRescanResult)
    df =  dfAll[namesOfColumn4Learning]

    numberOfTruePredictList = []
    for num in range(0,times):
        dfForBuildModel = df.sample(frac=0.8)
        dfFinalTest = df.loc[~df.index.isin(dfForBuildModel.index)]

        #dfForBuildModel, dfFinalTest  = model_selection.train_test_split(df, test_size=0.2, random_state = num*2)

        NormlizeColumnsValue(dfForBuildModel, dfFinalTest,namesOfColumn4Normliaze,3)
    
        len = df.shape[1]

        # Build data for the machine learning 

        array = dfForBuildModel.values
        data_x = array[:,1:len -1]
        data_y = array[:,len - 1]

        x_train, x_test, y_train, y_test   =  model_selection.train_test_split (data_x, data_y, test_size = 0.2, random_state = 42)

        #Start machine learning.

        model = GetMLModel(algorithmName)
        model.fit(x_train,y_train.astype(int))
        accuracy = model.score(x_test, y_test.astype(int))
        print("Accuracy = {}%".format(accuracy * 100))

        #Predict

        columnsOnlyX = dfFinalTest.columns[1:len-1]
        dfOnlyX = dfFinalTest[columnsOnlyX]
        dfFinalTest["Predict"] = model.predict(dfOnlyX)

        #Get specific data information

        lenOfFinalTest = dfFinalTest.shape[0] 
        lenOfDefinedPercentFinalTest = int(precision/100*lenOfFinalTest)
        dfFinalTest = dfFinalTest.sort_values(by="Predict")
        numberOfTruePredict = (dfFinalTest[nameOfRescanResult][dfFinalTest[nameOfRescanResult] > 0]).sum()
        numOfallPredict = df.shape[1]
        numberOfTruePredictList.append(numberOfTruePredict/len)

    ndList = np.array(numberOfTruePredictList)
    return ndList.mean()


   
