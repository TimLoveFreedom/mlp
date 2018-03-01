__author__ = 'jellyzhang'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlp import Mlp

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encode before splitting because matrix X and independent variable Y must be already encoded
# Found two categorical data (country, gender)
# create dummy variables, avoid dummy variable trap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])



mlp=Mlp(len(X[0]))
mlp.train(X,Y,epochs=10,batch_size=100)