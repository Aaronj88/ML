import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

hd = pd.read_csv("Housing Data.csv")

print(hd.isnull().sum())
hd.dropna(inplace=True)
print(hd.isnull().sum())
X = hd[["RM","LSTAT"]]
y = hd["MEDV"]




X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)

from sklearn.preprocessing import PolynomialFeatures
obj = PolynomialFeatures(degree=2)
X_train = obj.fit_transform(X_train)
X_test = obj.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

y_preds = model.predict(X_test)

'''plt.scatter(X_test["RM"],y_preds)
plt.show()'''

from sklearn.metrics import root_mean_squared_error
err = root_mean_squared_error(y_test,y_preds)
print(err)

print(X_train)



