# ------------------------------------------------------------
# 📚 Intelligent Book Recommender System (All-in-One Version)
# ------------------------------------------------------------
# 🧠 Auto-downloads book data (with images) using Google Books API
# 💡 Displays an interactive Streamlit UI with recommendations
# ------------------------------------------------------------
# 📦 Dependencies: streamlit pandas numpy scikit-learn requests
# 📦 Dependencies: streamlit pandas numpy scikit-learn requests
# 💻 Install once: pip install streamlit pandas numpy scikit-learn requests
# 🚀 Run app: streamlit run intelligent_book_recommender.py
# ------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="📚 Intelligent Book Recommender", layout="wide")

DATA_FILE_PATH = "merged_dataset.csv"

CATEGORY_KEYWORDS = {
    "Romance": ["love", "romance", "relationship", "affair"],
    "Horror/Thriller": ["horror", "thriller", "suspense", "fear", "murder"],
    "Psychology": ["psychology", "mind", "mental", "behavior"],
    "Adventure": ["adventure", "journey", "quest"],
    "Mystery": ["mystery", "detective", "crime"],
    "Fantasy": ["fantasy", "wizard", "dragon"],
    "Science Fiction": ["science", "future", "space", "alien"],
    "Self-Help": ["self-help", "motivation", "guide"],
}

# ---------------- DATA FETCH ----------------
@st.cache_data
def fetch_books(query):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=40"
    data = requests.get(url).json().get("items", [])

    books = []
    for b in data:
        info = b.get("volumeInfo", {})
        books.append({
            "Title": info.get("title", ""),
            "Author": ", ".join(info.get("authors", [])),
            "Published": info.get("publishedDate", ""),
            "Description": info.get("description", ""),
            "Ratings": info.get("averageRating", 0),
            "Image": info.get("imageLinks", {}).get("thumbnail", "")
        })
    return pd.DataFrame(books)

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE_PATH):
        df = pd.read_csv(DATA_FILE_PATH)
    else:
        topics = ["romance","science","fantasy","psychology","thriller","history"]
        df = pd.concat([fetch_books(t) for t in topics])
        df.drop_duplicates("Title", inplace=True)
        df.to_csv(DATA_FILE_PATH, index=False)

    df.fillna("", inplace=True)
    df["text"] = df["Title"] + " " + df["Author"] + " " + df["Description"]
    return df

@st.cache_data
def compute_similarity(text):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(text)
    return linear_kernel(matrix, matrix)

def detect_category(desc):
    desc = desc.lower()
    for cat, words in CATEGORY_KEYWORDS.items():
        if any(w in desc for w in words):
            return cat
    return "General"

# ---------------- UI ----------------
st.title("📚 Intelligent Book Recommender")
st.markdown("AI-powered personalized book suggestions")

df = load_data()
sim_matrix = compute_similarity(df["text"])

# Sidebar Trending
st.sidebar.title("🔥 Trending Books")
top = df.sort_values("Ratings", ascending=False).head(5)
for t in top["Title"]:
    st.sidebar.write("📘", t)

# Search
search = st.text_input("🔎 Search book name")
books = df["Title"].tolist()
if search:
    books = [b for b in books if search.lower() in b.lower()]

selected = st.selectbox("📚 Select Book", books)

min_rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.0)
category_filter = st.selectbox("📚 Category", ["All"] + list(CATEGORY_KEYWORDS.keys()))

if st.button("✨ Recommend"):
    idx = df[df["Title"] == selected].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:15]

    results = []
    for i, score in scores:
        row = df.iloc[i]
        if float(row["Ratings"] or 0) >= min_rating:
            results.append((row, score))

    st.subheader(f"📖 Recommendations for {selected}")

    for book, sim in results[:5]:
        cat = detect_category(book["Description"])
        if category_filter != "All" and cat != category_filter:
            continue

        col1, col2 = st.columns([1,4])
        with col1:
            if book["Image"]:
                st.image(book["Image"], use_container_width=True)
        with col2:
            st.markdown(f"### {book['Title']}")
            st.markdown(f"👤 {book['Author']}")
            st.markdown(f"⭐ Rating: {book['Ratings']}")
            st.markdown(f"📚 Category: {cat}")
            st.markdown(f"🔗 Similarity: {round(sim*100,2)}%")
            st.markdown(book["Description"][:300])
            st.divider()
