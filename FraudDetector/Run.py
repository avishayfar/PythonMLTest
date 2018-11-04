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
ResultsDf = pd.DataFrame(columns=['TestName', 'Model','Times','Precsion of Original Fraud', 'Precsion Of Predict Fruad','Original Precsion Of Lines For Recall','Precsion Of Lines For Recall', 'Original Shrinkage Per Line', 'Shrinkage Per Line'])
for i in range( len(inputDf.index)):
    testName = inputDf['TestName'].iloc[i]
    namesOfColumn4LearningLst = inputDf['namesOfColumn4Learning'].iloc[i].split()
    namesOfColumn4NormliazeLst = ""
    if(not pd.isnull(inputDf['namesOfColumn4Normliaze'].iloc[0])):
        namesOfColumn4NormliazeLst = inputDf['namesOfColumn4Normliaze'].iloc[i].split()
    dic = RunFraudDetector(inputDf['basicDataTableName'].iloc[i], inputDf['storeWithHierarchyTableName'].iloc[i], inputDf['modelName'].iloc[i], namesOfColumn4LearningLst, namesOfColumn4NormliazeLst, inputDf['RescanResultColumnName'].iloc[i], inputDf['ShrinkageColumnName'].iloc[i], inputDf['times'].iloc[i], inputDf['threshold'].iloc[i], inputDf['OriginRescanRate'].iloc[i])
    ResultsDf.loc[i] = [testName,dic["Model"],dic["Times"],dic["PrecsionOfOriginalFraud"], dic["PrecsionOfTruePredict"],dic["OriginalPrecisionOfLinesForRecall"],dic['PrecisionOfLinesForRecall'],dic['OriginalShrinkagePerLine'],dic['ShrinkagePerLine']]

#Save the results to file  
outputPath = "C:\\SeScFRD\\Results\\" + outputFileName 
ew = pd.ExcelWriter(outputPath)
ResultsDf.to_excel(ew, sheet_name='TestResult')
ew.save()







