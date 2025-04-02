from sklearn import datasets as ds
import pandas as pd

bc = ds.load_breast_cancer()
print(type(bc))
print(bc.keys())

X = pd.DataFrame(bc.data,columns = bc.feature_names)
y = pd.Series(bc.target)
print(X.head())
print(y.head())


print(X.info())
print(y.isnull())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)

from sklearn.svm import SVC
model = SVC(kernel="linear")

model.fit(X_train,y_train)
py = model.predict(X_test)

from sklearn.metrics import classification_report
err = classification_report(y_test,py)
print(err)

