"""
IG is univariate method->less training time->doesnt guarantee better accuracy
-->b/c calculating the accuracy individually.

Data preprocessing:
1)feature selection ->information gain attribute evaluation
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
# Importing the dataset
dataset = pd.read_csv('dermatology_csv.csv')
X = dataset.iloc[:, 0:34].values
y = dataset.iloc[:, 34].values



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
X= imputer.transform(X)

X=pd.DataFrame(X)



#calculate mi
mi = mutual_info_classif(X, y)

mi = pd.Series(mi)#One-dimensional ndarray with axis labels 
mi.index = X.columns

mi.sort_values(ascending=False, inplace = True)

mi.plot.bar(figsize = (16,5))

sel = SelectPercentile(mutual_info_classif, percentile=50).fit(X, y)
X.columns[sel.get_support()]
X.shape
X_mi = sel.transform(X)#Reduce X to the selected features.
X_mi.shape
#Build the model to compare the performance
def run_randomForest(X,y):
    #meta estimator that fits a number of decision tree
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print('Accuracy on mi set: ')
    print(accuracy_score(y, y_pred))
    


type(X_mi)
run_randomForest(X_mi,y)
print(X_mi.shape)



