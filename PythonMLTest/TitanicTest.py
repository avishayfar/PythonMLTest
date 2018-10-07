


import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC

# Just to switch off pandas warning
pandas.options.mode.chained_assignment = None

url = "C:\\ML\\CSV\\titanic_train.csv"
df = pandas.read_csv(url)
print(df.shape)
print(df.head(10))


median_age = df['age'].median()
print("Median age is {}".format(median_age))

df['age'].fillna(median_age, inplace = True)
print(df.head(10))


print()
data_inputs = df[["pclass", "age", "sex"]]
print(data_inputs.head(10))

print()
data_inputs["pclass"].replace("3rd", 3, inplace = True)
data_inputs["pclass"].replace("2nd", 2, inplace = True)
data_inputs["pclass"].replace("1st", 1, inplace = True)
print(data_inputs.head(10))

print()
data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
print(data_inputs.head(10))

print()
expected_output = df[["survived"]]
print(expected_output.head(10))


x_train, x_test, y_train, y_test   =  model_selection.train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)

print()
print(x_train.head())
print()
print(y_train.head())

#Start machine learning.

rf = RandomForestClassifier (n_estimators=100)
rf.fit(x_train, y_train)
accuracy = rf.score(x_test, y_test)
print("Accuracy = {}%".format(accuracy * 100))




