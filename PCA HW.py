#reading the data
import pandas as pd
t = pd.read_csv("titanic.csv")
print(t.isnull().sum()) #no null values


#cleaning data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
t.drop("Name",axis=1,inplace=True)
a = le.fit_transform(t["Sex"])
t["Sex"] = a


#splitting features and target
X = t.drop("Survived",axis=1)
y = t["Survived"]


#scaling (have to for PCA)
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X = mms.fit_transform(X)


#splitting into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)


#PCA
from sklearn.decomposition import PCA
pca = PCA(5)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
X_train = pd.DataFrame(X_train,columns=["PC1","PC2","PC3","PC4","PC5"])
print(X_train)


#making predictions (survived or not) with K_NN
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train,y_train)
py = model.predict(X_test)


#checking the accuracy
from sklearn.metrics import root_mean_squared_error
err = root_mean_squared_error(y_test,py)
print(err)


