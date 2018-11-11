import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


tempPath = "C:\\SeScFRD\\Input\\ItamarData.xlsx"
df = pd.read_excel(tempPath)

df.plot.scatter(x='TotalShrinkage',y='RescanResult',c='red',s=5)
plt.show()

##Save the results to file  
#outputPath = "C:\\SeScFRD\\Input\\avishayData.xlsx" 
#ew = pd.ExcelWriter(outputPath)
#ResultsDf.to_excel(ew, sheet_name='data')
#ew.save()