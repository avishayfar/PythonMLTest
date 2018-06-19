



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
dataset = pandas.read_csv(url)
print(dataset.shape)
print(dataset.head(10))


median_age = dataset['age'].median()
print("Median age is {}".format(median_age))

dataset['age'].fillna(median_age, inplace = True)
print(dataset.head(10))


print()
data_inputs = dataset[["pclass", "age", "sex"]]
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
expected_output = dataset[["survived"]]
print(expected_output.head(10))

#test_size - test_size=0.33 means 33% of the sample is to be used for testing, the other for training.
#random_state - is used to initialise the inbuilt randomiser, so we get the same result from the randomiser each time.
#in most machine learning the input called - X
#                         the expected output called - Y
inputs_train, inputs_test, expected_output_train, expected_output_test   =  model_selection.train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)

print()
print(inputs_train.head())
print()
print(expected_output_train.head())

#Start machine learning.

rf = RandomForestClassifier (n_estimators=100)
rf.fit(inputs_train, expected_output_train)
accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))




