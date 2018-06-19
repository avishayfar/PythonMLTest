o# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "C:\FutureSunday\Iris.data.txt"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# class distribution
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
#dataset.hist()
#plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# Split-out validation dataset
array = dataset.values
data_inputs = array[:,0:4]
expected_output = array[:,4]

#print data_inputs
print("data_inputs")
print(data_inputs)

#print expected_output
print("expected_output")
print(expected_output)


#test_size - test_size=0.33 means 33% of the sample is to be used for testing, the other for training.
#random_state - is used to initialise the inbuilt randomiser, so we get the same result from the randomiser each time.
#in most machine learning the input called - X
#                         the expected output called - Y
x_train,x_test, y_train, y_test = model_selection.train_test_split(data_inputs, expected_output, test_size=0.20, random_state=7)


# Test options and evaluation metric
seed = 7

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

 # Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print()
print("accuracy_score:")
print(accuracy_score(y_test, predictions))
print()
print("confusion_matrix:")
print(confusion_matrix(y_test, predictions))
print()
print("classification_report:")
print(classification_report(y_test, predictions))

print("predictions:")
print(accuracy_score(predictions))

print("y_test:")
print(accuracy_score(print("y_test:")
))