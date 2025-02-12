import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


t = pd.read_csv("titanic.csv")
t.drop("Name",axis=1,inplace=True)
print(t.isnull().sum())

print(t.info())

le = LabelEncoder()
a = le.fit_transform(t["Sex"])
t["Sex"] = a

print(t.corr())
import matplotlib.pyplot as plt
import seaborn as sb

sb.heatmap(t.corr(),annot=True)
plt.show()


X = t.drop("Survived",axis=1)
y = t["Survived"]

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)

py = model.predict(X_test)

print(py)


from sklearn.metrics import confusion_matrix #finding error

err = confusion_matrix(y_test,py)
print(err)

sb.heatmap(err,annot=True)
plt.xlabel("Predicted Results")
plt.ylabel("Actual Results")
plt.show()


from sklearn.metrics import classification_report #other way of finding error

err2 = classification_report(y_test,py)
print(err2)

