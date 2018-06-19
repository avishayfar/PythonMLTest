import os

arr = os.listdir('C:\\SeScFRD\\New\\')
for file in arr:
    fileResult = file[:-5] + "_Result.xsls"
    print (file + "   " + fileResult)