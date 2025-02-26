import pandas as pd

i = pd.read_csv("Iris.csv")
print(i.isnull().sum())

X = i.drop("species",axis=1)
y = i["species"]

#min-max scaling
from sklearn.preprocessing import MinMaxScaler
obj = MinMaxScaler()
X = obj.fit_transform(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=8,train_size=0.8)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
py = model.predict(X_test)


from sklearn.metrics import classification_report
err = classification_report(y_test,py)
print(err)


from sklearn.metrics import confusion_matrix
err2 = confusion_matrix(y_test,py)
print(err2)

print(X_train.shape[0])
rows = X_train.shape[0]

#finding optimal K
from sklearn.metrics import f1_score
import numpy as np
f1scores = []
for j in range(1,round(np.sqrt(rows))):
    model = KNeighborsClassifier(j,n_jobs = -1)
    model.fit(X_train,y_train)
    py = model.predict(X_test)
    f1scores.append(f1_score(y_test,py,average="macro"))


import matplotlib.pyplot as plt
plt.plot(range(1,round(np.sqrt(rows))),f1scores)
plt.show()

