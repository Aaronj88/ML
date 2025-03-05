import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

i = pd.read_csv("German_Cars.csv")
print(i.isnull().sum())
print(i.info())

import matplotlib.pyplot as plt
import seaborn as sb
sb.heatmap(i.corr(),annot=True)
plt.show()

print(i.columns)
X = i.drop("Price ",axis=1)
y = i["Price "]/10000


#min-max scaling
obj = MinMaxScaler()
X[["Year","Mileage", 'Fuel Consumption (L/100km)','Horsepower (HP)']] = obj.fit_transform(X[["Year","Mileage", 'Fuel Consumption (L/100km)','Horsepower (HP)']])


le = LabelEncoder()
X['Brand'] = le.fit_transform(X['Brand'])
X['Model'] = le.fit_transform(X['Model'])
X['Fuel Type'] = le.fit_transform(X['Fuel Type'])
X['Transmission'] = le.fit_transform(X['Transmission'])
X['City'] = le.fit_transform(X['City'])

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=8,train_size=0.8)


model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,y_train)
py = model.predict(X_test)


from sklearn.metrics import root_mean_squared_error
err = root_mean_squared_error(y_test,py)
print(err)



