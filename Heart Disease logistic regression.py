import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sb

h = pd.read_csv("Heart.csv")
print(h.head())
#0 null

X = h.drop("target",axis=1)
y = h["target"]


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=8)

m = LogisticRegression()
m.fit(X_train,y_train)

pred_y = m.predict(X_test)

from sklearn.metrics import classification_report
err = classification_report(y_test,pred_y)
print(err)

err2 = confusion_matrix(y_test,pred_y)
sb.heatmap(err2,annot=True,fmt="d")
plt.show()

