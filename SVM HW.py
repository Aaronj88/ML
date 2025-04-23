#reading data
import pandas as pd
h = pd.read_csv("Heart.csv")
print(h.isnull().sum()) #no null values

#splitting data
X = h.drop("target",axis=1)
y = h["target"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)

from sklearn.svm import SVC
model = SVC(kernel="linear")

model.fit(X_train,y_train)
py = model.predict(X_test)

from sklearn.metrics import classification_report
err = classification_report(y_test,py)
print(err)

