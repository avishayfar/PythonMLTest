
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier


def GetMLModel(algorithName):
    if (algorithName == 'LR'):
        return LogisticRegression()
    if (algorithName == 'LDA'):
        return LinearDiscriminantAnalysis()
    if (algorithName == 'KNN'):
        return KNeighborsClassifier()
    if (algorithName == 'RF'):
        rf = RandomForestClassifier (n_estimators=100)
    if (algorithName == 'CART'):
        return DecisionTreeClassifier()
    if (algorithName == 'GNB'):
        return GaussianNB()
    if (algorithName == 'SVC'):
        return SVC()
    if (algorithName == 'MLP'):
        return MLPClassifier(early_stopping=True,validation_fraction=0.2,max_iter=500)