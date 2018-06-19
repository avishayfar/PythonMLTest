
import pandas

# Just to switch off pandas warning
pandas.options.mode.chained_assignment = None

columns2delete =  np.array( ['ItemSaleAmount','ItemSaleQty','ItemReturnAmoun','ItemReturnQty','ItemSalePriceOverrideAmount','ItemSalePriceOverrideQty','ItemReturnPriceOverrideAmount','ItemReturnPriceOverrideQty'] )

url = "C:\\SeScFRD\\OnlyRescanData.xlsx"

df = pandas.read_excel(url)

#for col in columns2delete:
#    df = df.drop(col, 1)

df['Flag']=df['Flag'].fillna(0)

df.to_excel('C:\\FRD\\FRDTransaction.xlsx', sheet_name='OriginalValues')