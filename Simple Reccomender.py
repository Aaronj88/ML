import pandas as pd

df = pd.read_csv("Movies Metadata.csv")
print(df.info())
print(df.head())


'''
Weighted Rating - 
(v/(v+m))*r + (m/(v+m))*c
v is the number of votes (vote_count)
m is the minimum votes required to be listed in chart
r is the average rating for the movie (vote_average)
c is the mean vote across the whole report
'''

v = df["vote_count"]
m = v.quantile(0.90)
r = df["vote_average"]
c = r.mean()

wr = (v/(v+m))*r+(m/(v+m))*c
df["weighted_rating"]=wr

df = df[["title","vote_average","vote_count","weighted_rating"]]
print(df.head(10))

df_sorted = df.sort_values("weighted_rating",ascending=False)
print(df_sorted.head(20))



