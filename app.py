import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("movies.csv")
df = df.dropna(subset=['MOVIES','GENRE','ONE-LINE'])
df['combined'] = df['GENRE'].astype(str) + " " + df['ONE-LINE'].astype(str) + " " + df['STARS'].astype(str)
df['MOVIES_norm'] = df['MOVIES'].str.strip().str.lower()


vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(df['combined'])
cosine_sim = cosine_similarity(matrix, matrix)
indices = pd.Series(df.index, index=df['MOVIES']).drop_duplicates()


def recommend_multiple(selected_titles, num_recs=10):
    import numpy as np

    selected_norm = [t.strip().lower() for t in selected_titles]

    sim_scores = [cosine_sim[indices[movie]] for movie in selected_titles if movie in indices]
    if not sim_scores:
        return []

    avg_sim = np.mean(sim_scores, axis=0)
    scores = list(enumerate(avg_sim))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended_titles = []
    for idx, score in scores:
        title = df.iloc[idx]['MOVIES']
        title_norm = df.iloc[idx]['MOVIES_norm']
        if title_norm not in selected_norm and title not in recommended_titles:
            recommended_titles.append(title)
        if len(recommended_titles) >= num_recs:
            break

    return recommended_titles







st.title("🎬 Movie Recommendation System")

selected_movies = st.multiselect(
    "Pick between 1–5 of your favorite movies:",
    df['MOVIES'].tolist(),
    max_selections=5
)


if st.button("Get Recommendations"):
    if len(selected_movies) == 0:
        st.warning("Please select at least one movie.")
    else:
        recs = recommend_multiple(selected_movies)
        st.subheader("Recommended Movies:")
        for i, rec in enumerate(recs, start=1):
            st.write(f"{i}. {rec}")
