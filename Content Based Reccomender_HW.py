import pandas as pd

#reading/cleaning the data
df = pd.read_csv("songs.csv")
print(df.info())
df = df[["Description","Name of the Song"]]
print(df.isnull().sum())
df.dropna(inplace=True) #4369 of 198127 null
print(df.isnull().sum())
titles = pd.Series(df.index,index=df["Name of the Song"]).drop_duplicates()
print(df["Description"].head())


#getting rid of stop words
from sklearn.feature_extraction.text import TfidfVectorizer #Term Frequency 
obj = TfidfVectorizer(stop_words="english")
dfidf_matrix = obj.fit_transform(df["Description"])
print(dfidf_matrix.shape)
print(obj.get_feature_names_out()[5000:5010])

#finding the similarity 
from sklearn.metrics.pairwise import linear_kernel
similarity = linear_kernel(dfidf_matrix,dfidf_matrix)

def finder(name):
    ind = titles[name]
    sim_scores = list(enumerate(similarity[ind]))
    print(sim_scores[:10])
    sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)
    top_ten = sim_scores[:10]
    print(top_ten)
    movie_index = [item[0] for item in top_ten]
    return df[["Name of the Song"]].iloc[movie_index]


finder("Nevermind [20th Anniversary Edition]")




