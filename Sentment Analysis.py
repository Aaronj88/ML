import pandas as pd

txt = pd.read_csv("sentiments.txt",names=["text","sentiment"],sep=";")
print(txt)
print(txt["sentiment"].value_counts())

txt["sentiment"] = txt["sentiment"].replace({"joy":1,"love":1,"surprise":1,"anger":0,"fear":0,"sadness":0})
print(txt["sentiment"].value_counts())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train_y_test = train_test_split(txt["text"],txt["sentiment"],train_size=0.8,random_state=8)

import nltk
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

lemmatizer = WordNetLemmatizer()
def text_transform(data):
    corpus = []
    for sentence in data:
        item = re.sub('[^a-zA-Z]'," ",sentence)
        item = item.lower()
        item = item.split(" ")
        lema = []
        for word in item:
            if word not in stopwords.words("english"):
                lema.append(lemmatizer.lemmatize(word))







