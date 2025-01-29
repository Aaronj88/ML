from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sdf = pd.read_csv('Salaries.csv')
print(sdf.info())
print(sdf.isnull().sum())


y = sdf["Salary"]
X = sdf["YearsExperience"]
X = np.array(X)



X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)

X_train =  X_train.reshape(-1,1)
X_test =  X_test.reshape(-1,1)

model = LinearRegression()
model.fit(X_train,y_train)
pred_y = model.predict(X_test)

error = root_mean_squared_error(y_test,pred_y)
print(error)

plt.scatter(X_test,y_test)
plt.plot(X_test,pred_y,"g-o")
plt.show()


