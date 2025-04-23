import pandas as pd

df = pd.read_csv("Movies Metadata.csv")
print(df.info())
df = df[["overview","title"]]
print(df["overview"].head())
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())
print(df.shape)


from sklearn.feature_extraction.text import TfidfVectorizer #Term Frequency 
obj = TfidfVectorizer(stop_words="english")
dfidf_matrix = obj.fit_transform(df["overview"])

print(dfidf_matrix.shape)
print(obj.get_feature_names_out()[5000:5010])

#finding the similarity
from sklearn.metrics.pairwise import linear_kernel
similarity = linear_kernel(dfidf_matrix,dfidf_matrix)
print(similarity.shape)
print(similarity[:10,:10])

titles = pd.Series(df.index,index=df["title"]).drop_duplicates()

def finder(name):
    ind = titles[name]
    sim_scores = enumerate(similarity[ind])
    print(sim_scores[:10])
    sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)
    top_ten = sim_scores[:10]
    print(top_ten)
    movie_index = [item[0] for item in top_ten] #list comprehension
    return df[["titles"]].iloc[movie_index]


finder("The Dark Knight Rises")






