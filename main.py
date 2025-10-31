import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("movies.csv")



df = df.dropna(subset=['MOVIES','GENRE','ONE-LINE']) 

df['combined'] = df['GENRE'].astype(str) + " " + df['ONE-LINE'].astype(str) + " " + df['STARS'].astype(str)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['MOVIES']).drop_duplicates()

def recommend(movie_title, num_recs=5):
    if movie_title not in indices:
        return f"Sorry — '{movie_title}' not found in dataset."
    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
 
    sim_scores = sim_scores[1 : 1 + num_recs]

    movie_indices = [i[0] for i in sim_scores]
    return df['MOVIES'].iloc[movie_indices].tolist()

print(recommend("Blood Red Sky", num_recs=5))
