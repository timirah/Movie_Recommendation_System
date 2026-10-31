from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

# the models folder sits next to this file
MODELS = Path(__file__).parent / "models"


# ---------- 1. Load everything once ----------
@st.cache_resource
def load_everything():
    svd = joblib.load(MODELS / "svd.pkl")                      # Model A
    matrix = load_npz(MODELS / "matrix.npz")                   # movie x user table (Model B)
    movie_ids = joblib.load(MODELS / "movie_ids.pkl")          # row number -> movie ID
    movies = pd.read_csv(MODELS / "movies.csv").set_index("movieId")
    seen = (pd.read_csv(MODELS / "seen.csv")
            .groupby("userId")["movieId"].apply(set).to_dict())  # movies each user already rated

    # the similar-rows tool is rebuilt here; fitting only stores the rows, so it is instant
    knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(matrix)
    row_of = {m: i for i, m in enumerate(movie_ids)}           # movie ID -> row number
    return svd, matrix, movie_ids, row_of, movies, seen, knn


svd, matrix, movie_ids, row_of, movies, seen, knn = load_everything()


# ---------- 2. The two recommenders ----------
def similar_movies(movie_id, n=10):
    """Model B: movies whose rating patterns look like this movie's."""
    distances, rows = knn.kneighbors(matrix[row_of[movie_id]], n_neighbors=n + 1)
    ids = [movie_ids[r] for r in rows[0][1:]]                  # skip the movie itself
    result = movies.loc[ids, ["title", "genres"]].copy()
    result["similarity"] = (1 - distances[0][1:]).round(3)
    return result.reset_index(drop=True)


def recommend_for_user(user_id, n=10):
    """Model A: the movies this user is predicted to rate highest."""
    already = seen.get(user_id, set())
    candidates = [m for m in movies.index if m not in already]
    guesses = [(m, svd.predict(user_id, m).est) for m in candidates]
    guesses.sort(key=lambda pair: pair[1], reverse=True)
    top = guesses[:n]
    result = movies.loc[[m for m, _ in top], ["title", "genres"]].copy()
    result["predicted_rating"] = [round(r, 2) for _, r in top]
    return result.reset_index(drop=True)


# ---------- 3. The web page ----------
st.title("Movie Recommender")
st.write("Two collaborative filtering models trained on the MovieLens ratings.")

tab_similar, tab_user = st.tabs(["Find Me Movies Similar to this One", "Recommend Movies for a User"])

with tab_similar:
    st.write("Pick a movie you like and see which movies have similar rating patterns.")
    options = movies.sort_values("title").index.tolist()
    movie_id = st.selectbox(
        "Pick a movie",
        options,
        format_func=lambda m: movies.loc[m, "title"],
        key="movie_pick",
    )
    n_similar = st.slider("How many?", 5, 20, 10, key="n_similar")
    if st.button("Find similar movies", key="go_similar"):
        st.dataframe(similar_movies(movie_id, n_similar), hide_index=True)

with tab_user:
    st.write("Enter a user ID to see what the model predicts they would rate highest.")
    user_id = st.number_input(
        "User ID",
        min_value=int(min(seen)),
        max_value=int(max(seen)),
        value=1,
        step=1,
    )
    n_user = st.slider("How many?", 5, 20, 10, key="n_user")
    if st.button("Recommend", key="go_user"):
        st.dataframe(recommend_for_user(int(user_id), n_user), hide_index=True)
    st.caption(
        "Predicted ratings near 5 are a ranking, not a promise. "
        "The tuned model leans toward movies that are well liked by most people."
    )