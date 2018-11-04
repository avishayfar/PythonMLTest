import numpy as np
import pandas as pd


ResultsDf = pd.DataFrame(columns=['NumberOfItems','NumberOfVoidedItems', 'RescanResult'])
for i in range(0, 1000):
    NumberOfItems = i
    NumberOfVoidedItems = i
    RescanResult = 1 if i>500 else 0
    ResultsDf.loc[i] = [NumberOfItems, NumberOfVoidedItems, RescanResult]

#Save the results to file  
outputPath = "C:\\SeScFRD\\Input\\avishayData.xlsx" 
ew = pd.ExcelWriter(outputPath)
ResultsDf.to_excel(ew, sheet_name='data')
ew.save()