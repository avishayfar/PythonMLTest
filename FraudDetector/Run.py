from FraudDetector import RunFraudDetector
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

outputFileName = "TestResult1.xlsx"
inputFilePath = "C:\SeScFRD\Input\InputParameters.xlsx"

#Read the Input file
inputDf = pd.read_excel(inputFilePath)

#Run the fraud detector on the input data
ResultsDf = pd.DataFrame(columns=['TestName', 'Result'])
for i in range( len(inputDf.index)):
    testName = inputDf['TestName'].iloc[i]
    namesOfColumn4LearningLst = inputDf['namesOfColumn4Learning'].iloc[i].split()
    namesOfColumn4NormliazeLst = inputDf['namesOfColumn4Normliaze'].iloc[i].split()
    result = RunFraudDetector(inputDf['basicDataTableName'].iloc[i], inputDf['storeWithHierarchyTableName'].iloc[i], inputDf['modelName'].iloc[i], namesOfColumn4LearningLst, namesOfColumn4NormliazeLst, inputDf['nameOfRescanResult'].iloc[i], inputDf['times'].iloc[i], inputDf['threshold'].iloc[i], inputDf['precision'].iloc[i])
    ResultsDf.loc[i] = [testName,result]

#Save the results to file  
outputPath = "C:\\SeScFRD\\Results\\" + outputFileName 
ew = pd.ExcelWriter(outputPath)
ResultsDf.to_excel(ew, sheet_name='TestResult')
ew.save()








#basicDataTableName = "ItamarBaseData"
#storeWithHierarchyTableName = "StoreWithHierarchy"
#modelName =  'MLP'
#namesOfColumn4Learning = ['NumberOfItems','TransactionSSCTotal','AvgItemPrice','ItemPriceVariation','TrustLevel1','TrustLevel2','TrustLevel3','TrustLevel4','TrustLevel5','NumberOfVoidedItems','ValueOfVoidedLines','AvgVoidedItemPrice','NumberOfDistinctItem','RatioOfDistinctItem','transactionCloseTime','TransactionScanTimeDuration']
#namesOfColumn4Normliaze  =['NumberOfItems','TransactionSSCTotal','AvgItemPrice','ItemPriceVariation','NumberOfVoidedItems','ValueOfVoidedLines', 'AvgVoidedItemPrice','NumberOfDistinctItem','RatioOfDistinctItem','TransactionScanTimeDuration']
#nameOfRescanResult = 'RescanResult'
#precision = 15
#threshold = 3 
#times = 10
