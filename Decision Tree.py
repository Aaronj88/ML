import pandas as pd

c = pd.read_csv("Cars.csv")
print(c.isnull().sum()) #no null
print(c.info())

print(c["class"].value_counts())

print(c["doors"].value_counts()," ",c["persons"].value_counts())

c["doors"].replace({"5more":5},inplace=True)
c["persons"].replace({"more":7},inplace=True)
print(c["doors"].value_counts()," ",c["persons"].value_counts())


c = c.astype({"doors":int,"persons":int})
print(c.info())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c["sales"] = le.fit_transform(c["sales"])
c["maintenance"] = le.fit_transform(c["maintenance"])
c["boot_space"] = le.fit_transform(c["boot_space"])
c["safety"] = le.fit_transform(c["safety"])
c["class"] = le.fit_transform(c["class"])

print(c.info())
print(c["sales"].value_counts())
print(c["maintenance"].value_counts())
print(c["boot_space"].value_counts())
print(c["safety"].value_counts())
print(c["class"].value_counts())

X = c.drop("class",axis=1)
y = c["class"]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=8)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
py = model.predict(X_test)



from sklearn.metrics import confusion_matrix
err = confusion_matrix(y_test,py)
print(err)

from sklearn.metrics import classification_report
err2 = classification_report(y_test,py)
print(err2)



print(model.predict([[3,0,1,2,1,2]]))
print(y.value_counts())