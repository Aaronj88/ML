import pandas as pd

ad_in = pd.read_csv("adult_income.csv",sep=", ")
print(ad_in.isnull().sum()) #no null

print(ad_in.info())
print(ad_in.head())

print(ad_in["income"].value_counts())

X = ad_in.drop(["fnlwgt","education","income"],axis=1)
y = ad_in["income"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X["workclass"] = le.fit_transform(X["workclass"])
X["marital-status"] = le.fit_transform(X["marital-status"])
X["occupation"] = le.fit_transform(X["occupation"])
X["relationship"] = le.fit_transform(X["relationship"])
X["race"] = le.fit_transform(X["race"])
X["gender"] = le.fit_transform(X["gender"])
X["native-country"] = le.fit_transform(X["native-country"])
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)


#importing the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
py = model.predict(X_test)


from sklearn.metrics import classification_report
err = classification_report(y_test,py)
print(err)

from sklearn.metrics import confusion_matrix
err2 = confusion_matrix(y_test,py)
print(err2)

