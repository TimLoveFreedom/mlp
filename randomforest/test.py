__author__ = 'jellyzhang'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('../Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


split_point=math.ceil(len(X)*0.8)
test_X,test_Y=X[:split_point],Y[:split_point]
valid_X,valid_Y=X[split_point:],Y[split_point:]
#归一化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test_X = sc.fit_transform(test_X)
valid_X = sc.transform(valid_X)
test_Y=np.transpose(test_Y[None,:])
valid_Y=np.transpose(valid_Y[None,:])

#train
rf=RandomForestClassifier(n_estimators=60,oob_score=True,random_state=10)
rf.fit(test_X,test_Y)
#predict
predictions=rf.predict(valid_X)
print(metrics.precision_score(valid_Y, predictions))   #0.734


