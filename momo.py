import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
import warnings

warnings.filterwarnings('ignore')

# import two datasets
df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

# merge two dataset
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

df2.head(5)

df2.shape
df2.info()

df2['release_date'] = pd.to_datetime(df2['release_date'])
# check if there's missing value
df2.isnull().sum()

df2['overview'] = df2['overview'].fillna('') # fill with empty string
df2['runtime'] = df2['runtime'].fillna(df2['runtime'].median()) # fill with median value
df2 = df2.dropna(subset=['release_date']) # since there's only one missing value, just remove it
df2 = df2.drop('homepage', axis=1)
df2['tagline'] = df2['tagline'].fillna('')

# check columns exist
df2[['title', 'vote_average', 'vote_count']].head()

df2.isnull().sum()

# look at some statistics
df2[['vote_average', 'vote_count']].describe()

# change the text to number, make it easy to compare
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape
