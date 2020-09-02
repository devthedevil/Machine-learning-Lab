# -*- coding: utf-8 -*-
"""
Data preprocessing:
2)Feature forward method & Feature backward method
"""

!pip install mlxtend

# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
# Importing the dataset
dataset = pd.read_csv('dermatology_csv.csv')
X = dataset.iloc[:, 0:34].values
y = dataset.iloc[:, 34].values



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
X= imputer.transform(X)
imputer = imputer.fit(X)
X= imputer.transform(X)

X=pd.DataFrame(X)

#Step Forward selection
#RandomForestClassifier for performance
sfs = SFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs = -1),
         k_features = 4,
          forward= True,
          floating = False,
          verbose= 2,#score and time
          scoring= 'accuracy',
          cv = 4,#divide by 4 and take average of 4 for accuracy
          n_jobs= -1
         ).fit(X, y)

sfs.k_feature_names_
sfs.k_feature_idx_

sfs.k_score_
pd.DataFrame.from_dict(sfs.get_metric_dict()).T

sfs = SFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs = -1),
         k_features = (1, 8),
          forward= True,
          floating = False,
          verbose= 2,
          scoring= 'accuracy',
          cv = 4,
          n_jobs= -1
         ).fit(X, y)

sfs.k_score_
sfs.k_feature_names_
###Step Backward Selection (SBS)

sfs = SFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs = -1),
         k_features = (1, 8),
          forward= False,
          floating = False, 
          verbose= 2,
          scoring= 'accuracy',
          cv =.gvyy 4,
          n_jobs   ,.mm,m,= -1
         ).fit(X, y)


sbs = sfs
sbs.k_score_
sbs.k_feature_names_

