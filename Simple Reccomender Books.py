#reading data
import pandas as pd
df = pd.read_csv("books.csv")


#removing null values
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())


#calculating weighted rating
print(df.info())
v = df["Book_ratings_count"]
m = v.quantile(0.99)
r = df["Book_average_rating"]
c = r.mean()

wr = (v/(v+m))*r+(m/(v+m))*c
df["weighted_rating"]=wr


#sorting the data so the highest rated are at the top
df = df[["Book Name","Book_average_rating","Book_ratings_count","weighted_rating"]]
df_sorted = df.sort_values("weighted_rating",ascending=False)
print("TOP TEN HIGHEST RATED BOOKS:")
print(df_sorted.head(10))