import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hd = pd.read_csv("Housing Data.csv")
print(hd.info())
hd = hd["RM","LSTAT","MEDV"]

y = hd["MEDV"]
X = hd["LSTAT","RM"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)



