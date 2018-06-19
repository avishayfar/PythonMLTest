import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#dataset.to_excel('C:\FutureSunday\stam.xlsx', sheet_name='Sheet1')

matrix = np.arange(40).reshape(10,4)
column = np.array( ["x1","x2","x3","x4"])

df = pd.DataFrame(matrix, columns=column)

print(df)

df["y"] = np.arange(10)

print("with y")

print(df)

df5 = df[df["y"] > 5]

print ("df5")

print (df5)

#f5.to_excel('C:\FutureSunday\stam.xlsx', sheet_name='Sheet1')

print("columnsOnlyX")
columnsOnlyX = df.columns[:4]
print(df[columnsOnlyX])

print("X1")
x1 = df["x1"] 
print(x1)

df["fraud"] = x1

print("df + fraud")
print(df)