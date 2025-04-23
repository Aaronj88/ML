import pandas as pd

df = pd.read_csv("Iris.csv")
print(df.isnull().sum()) #no null values

X = df.drop("species",axis=1)
y = df["species"]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=8)


#decision tree
from sklearn.tree import DecisionTreeClassifier
m = DecisionTreeClassifier()
m.fit(X_train,y_train)
pred_dt = m.predict(X_test)


#random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
pred_rf = model.predict(X_test)



from sklearn.metrics import classification_report
print("DECISION TREE REPORT:")
err = classification_report(pred_dt,y_test)
print(err)

print("RANDOM FOREST REPORT:")
err2 = classification_report(pred_rf,y_test)
print(err2)


