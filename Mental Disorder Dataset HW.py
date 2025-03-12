import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import confusion_matrix


le = LabelEncoder()

md = pd.read_csv("Mental Disorders.csv")
print(md.isnull().sum()) #no null
print(md)


#fit transform
md['Sadness'] = le.fit_transform(md['Sadness'])
md['Euphoric'] = le.fit_transform(md['Euphoric'])
md['Exhausted'] = le.fit_transform(md['Exhausted'])
md['Sleep Issues'] = le.fit_transform(md['Sleep Issues'])
md['Mood Swing'] = le.fit_transform(md['Mood Swing'])
md['Suicidal Thoughts'] = le.fit_transform(md['Suicidal Thoughts'])
md['Anorexia'] = le.fit_transform(md['Anorexia'])
md['Authority Respect'] = le.fit_transform(md['Authority Respect'])
md['Aggressive Response'] = le.fit_transform(md['Aggressive Response'])
md['Ignore & Move-On'] = le.fit_transform(md['Ignore & Move-On'])
md['Nervous Break-down'] = le.fit_transform(md['Nervous Break-down'])
md['Admits Mistakes'] = le.fit_transform(md['Admits Mistakes'])
md['Overthinking'] = le.fit_transform(md['Overthinking'])
md['Expert Diagnosis'] = le.fit_transform(md['Expert Diagnosis'])


X = md.drop(["Patient Number","Concentration","Optimism","Expert Diagnosis"],axis=1)
y = md["Expert Diagnosis"]


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=8,train_size=0.8)



m = LogisticRegression()
m.fit(X_train,y_train)

pred_y = m.predict(X_test)


from sklearn.metrics import classification_report
err = classification_report(y_test,pred_y)
print("logistic regression",err)


from sklearn.neighbors import KNeighborsClassifier
#finding optimal K
rows = X_train.shape[0]
from sklearn.metrics import f1_score
import numpy as np
f1scores = []
for j in range(1,round(np.sqrt(rows))):
    model = KNeighborsClassifier(j,n_jobs = -1)
    model.fit(X_train,y_train)
    py = model.predict(X_test)
    f1scores.append(f1_score(y_test,py,average="macro"))


from sklearn.metrics import classification_report
err2 = classification_report(y_test,py)
print(err2)


