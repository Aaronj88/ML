from sklearn import datasets as ds
import pandas as pd

da_s = ds.load_breast_cancer()
print(da_s.keys())
X = pd.DataFrame(da_s.data,columns=da_s.feature_names)
y = pd.Series(da_s.target)

print(X.info())


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,train_size=0.8,random_state=8)

#PCA
from sklearn.decomposition import PCA
pca = PCA(5)
X_pca = pca.fit_transform(X_train)
X_pca_test = pca.fit_transform(X_test)

print(X_pca)

X_c = pd.DataFrame(X_pca,columns=["PC1","PC2","PC3","PC4","PC5"])
print(X_c)


from sklearn.tree import DecisionTreeClassifier
m = DecisionTreeClassifier()
m.fit(X_pca,y_train)
py = m.predict(X_pca_test)


from sklearn.metrics import confusion_matrix
err = confusion_matrix(y_test,py)
print(err)

from sklearn.metrics import classification_report
err2 = classification_report(y_test,py)
print(err2)



#optimal number of principal components
from sklearn.metrics import f1_score
errs = []
for i in range(1,9):
    pca = PCA(i)
    op_pca = pca.fit_transform(X_train)
    op_pca_test = pca.fit_transform(X_test)
    model_op = DecisionTreeClassifier()
    model_op.fit(op_pca,y_train)
    py = model_op.predict(op_pca_test)
    err = f1_score(y_test,py,average="macro")
    errs.append(err)


op_pca = errs.index(max(errs))+1
print(errs)
print(op_pca)




    




