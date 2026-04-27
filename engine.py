import pandas as pd
import numpy as np
import ast, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from thefuzz import process

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_IMG     = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH  = "https://api.themoviedb.org/3/search/movie"

@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv").head(1500)
    cols = ['title','overview','genres','keywords','vote_average','release_date','popularity','vote_count']
    df = df[cols].dropna()
    def names(col):
        try: return " ".join([i['name'] for i in ast.literal_eval(col)])
        except: return ""
    df['genres_clean']   = df['genres'].apply(names)
    df['keywords_clean'] = df['keywords'].apply(names)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    df['soup'] = df['overview'] + " " + df['genres_clean'] + " " + df['keywords_clean']
    return df.reset_index(drop=True)

@st.cache_data
def build_tfidf_sim(df):
    tfidf  = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
    matrix = tfidf.fit_transform(df['soup'])
    return cosine_similarity(matrix, matrix)

@st.cache_data
def build_collab_sim(df):
    scaler   = MinMaxScaler()
    features = scaler.fit_transform(df[['vote_average','popularity','vote_count']].fillna(0))
    return cosine_similarity(features, features)

def hybrid_score(title, df, csim, psim, weights, n=12):
    indices = pd.Series(df.index, index=df['title'])
    if title not in indices: return pd.DataFrame()
    idx = indices[title]

    scaler   = MinMaxScaler()
    max_year = df['year'].max()

    content_s  = csim[idx]
    pop_s      = scaler.fit_transform(df[['popularity']].fillna(0)).flatten()
    rating_s   = scaler.fit_transform(df[['vote_average']].fillna(0)).flatten()
    recency_s  = np.where(df['year'] > 0, (df['year'] - df['year'].min()) / (max_year - df['year'].min() + 1), 0)

    # Diversity: penalize same genre as input
    input_genre = df.iloc[idx]['genres_clean']
    diversity_s = np.array([0.3 if g == input_genre else 1.0 for g in df['genres_clean']])

    w = weights
    total = w['content']+w['pop']+w['rating']+w['recency']+w['diversity']
    final = (
        w['content']  / total * content_s +
        w['pop']      / total * pop_s +
        w['rating']   / total * rating_s +
        w['recency']  / total * recency_s +
        w['diversity']/ total * diversity_s
    )
    final[idx] = 0  # exclude self

    result = df.copy()
    result['final_score']  = final
    result['content_s']    = content_s
    result['pop_s']        = pop_s
    result['rating_s']     = rating_s
    result['recency_s']    = recency_s
    result['diversity_s']  = diversity_s
    return result.nlargest(n, 'final_score')

def fuzzy_match(query, titles, limit=8):
    results = process.extract(query, titles, limit=limit)
    return [r[0] for r in results if r[1] > 45]

@st.cache_data(ttl=3600)
def get_poster(title, year=""):
    try:
        r = requests.get(TMDB_SEARCH, params={"api_key": TMDB_API_KEY, "query": title, "year": year}, timeout=4)
        res = r.json().get("results", [])
        if res and res[0].get("poster_path"):
            return TMDB_IMG + res[0]["poster_path"]
    except: pass
    return None