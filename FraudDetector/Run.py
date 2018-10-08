from FraudDetector import RunFraudDetector

namesOfColumn4Learning = ['TrxTotalNzd','VoidedItemsNzd','ValueVoidedLinesNzd','EarlyMorning','lunch','QuantitiyItemsNzd']
namesOfColumn4Normliaze  =['TrxTotalNzd','VoidedItemsNzd','ValueVoidedLinesNzd','QuantitiyItemsNzd']
nameOfRescanResult = 'RescanResult'
threshold = 3 



RunFraudDetector(namesOfColumn4Learning, namesOfColumn4Normliaze, nameOfRescanResult,threshold)
