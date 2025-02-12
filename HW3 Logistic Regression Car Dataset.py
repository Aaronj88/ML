import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler

c = pd.read_csv("Cars.csv")
print(c.head(10))
#no null values

c["doors"].replace("5more",5,inplace=True)
c["doors"] = c["doors"].astype(int)


X = c[["sales","maintenance","boot_space","safety"]]
y = c["class"]

X = pd.get_dummies(X,dtype="int")
X["doors"] = c["doors"]
print(X.info())

print(y.value_counts())

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)

print(y_train.value_counts())

r_u = RandomUnderSampler(sampling_strategy="not minority",random_state=8)
X_train,y_train = r_u.fit_resample(X_train,y_train)

print(y_train.value_counts())

model = LogisticRegression()
model.fit(X_train,y_train)

py = model.predict(X_test)

from  sklearn.metrics import confusion_matrix
err = confusion_matrix(y_test,py)
print(err)


import matplotlib.pyplot as plt
import seaborn as sb
sb.heatmap(err,annot=True,fmt='d')
plt.show()

from sklearn.metrics import classification_report
err2 = classification_report(y_test,py)
print(err2)

