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

# ------------------ CONFIGURATION ------------------
DATA_FILE_PATH = "merged_dataset.csv"
COLUMNS_TO_USE = ["Title", "Author", "Published", "Description", "Score", "Ratings", "Image"]
TEXT_FEATURE_COLUMNS = ["Title", "Author", "Description", "Published"]

CATEGORY_KEYWORDS = {
    "Romance": ["love", "romance", "relationship", "affair"],
    "Horror/Thriller": ["horror", "thriller", "suspense", "fear", "murder", "killer"],
    "Psychology": ["psychology", "mind", "mental", "behavior", "therapy"],
    "Adventure": ["adventure", "quest", "journey", "expedition"],
    "Mystery": ["mystery", "detective", "crime", "clue", "investigation"],
    "Fantasy": ["fantasy", "magical", "mythical", "wizard", "dragon"],
    "Historical Fiction": ["history", "historical", "past era", "period"],
    "Science Fiction": ["science", "scientific", "future", "space", "alien", "sci-fi"],
    "Biography/Memoir": ["biography", "memoir", "autobiography", "life story"],
    "Self-Help": ["self-help", "guide", "improvement", "motivation", "how to"],
    "Contemporary": ["contemporary", "modern life", "slice of life"],
    "Young Adult": ["young adult", "teen", "coming of age", "ya"]
}

# ------------------ STEP 1: FETCH DATA ------------------
@st.cache_data
def fetch_books_data(query="bestseller", max_results=30):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
    response = requests.get(url)
    books = response.json().get("items", [])
    
    data = []
    for book in books:
        info = book.get("volumeInfo", {})
        data.append({
            "Title": info.get("title", ""),
            "Author": ", ".join(info.get("authors", [])) if "authors" in info else "",
            "Published": info.get("publishedDate", ""),
            "Description": info.get("description", ""),
            "Score": info.get("ratingsCount", ""),
            "Ratings": info.get("averageRating", ""),
            "Image": info.get("imageLinks", {}).get("thumbnail", "")
        })
    return pd.DataFrame(data)

@st.cache_data
def build_dataset():
    topics = ["romance", "science fiction", "fantasy", "psychology", "thriller", "self-help", "history", "adventure"]
    all_books = pd.concat([fetch_books_data(t, 40) for t in topics], ignore_index=True)
    all_books.drop_duplicates(subset=["Title"], inplace=True)
    all_books.to_csv(DATA_FILE_PATH, index=False)
    return all_books

# ------------------ STEP 2: LOAD & PROCESS ------------------
@st.cache_data
def load_and_preprocess_data():
    if os.path.exists(DATA_FILE_PATH):
        df = pd.read_csv(DATA_FILE_PATH)
    else:
        st.info("📦 No local dataset found. Creating one now...")
        df = build_dataset()

    df.fillna("", inplace=True)
    available_features = [c for c in TEXT_FEATURE_COLUMNS if c in df.columns]
    df["text_features"] = df[available_features].astype(str).agg(" ".join, axis=1)
    df["text_features"] = df["text_features"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df

# ------------------ STEP 3: CALCULATE SIMILARITY ------------------
@st.cache_data
def calculate_similarity_matrix(text_series):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(text_series)
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    return sim_matrix

def extract_book_categories(description_text):
    text = (description_text or "").lower()
    matched = []
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched.append(cat)
                break
    return ", ".join(matched) if matched else "General"

def get_recommendations(title_query, df, similarity_matrix, count=5):
    indices = pd.Series(df.index, index=df["Title"]).drop_duplicates()
    matches = [t for t in indices.index if title_query.lower() == str(t).lower()]
    if not matches:
        st.warning("❌ Book not found. Try another title.")
        return pd.DataFrame()
    idx = indices[matches[0]]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: count + 1]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices]

# ------------------ STEP 4: STREAMLIT UI ------------------
st.set_page_config(page_title="📚 Intelligent Book Recommender", layout="wide")
st.title("📚 Intelligent Book Recommender System")
st.markdown("### Get personalized book suggestions based on your favorite titles!")

df = load_and_preprocess_data()
sim_matrix = calculate_similarity_matrix(df["text_features"])

# Dropdown for book titles
selected_book = st.selectbox("🔍 Choose a book you like:", sorted(df["Title"].unique()))

if st.button("✨ Recommend Books"):
    recommendations = get_recommendations(selected_book, df, sim_matrix, count=5)

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader(f"📖 Top Recommendations similar to **{selected_book}**:")
        for _, book in recommendations.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                if book["Image"]:
                    st.image(book["Image"], use_container_width=True)
            with col2:
                st.markdown(f"### {book['Title']}")
                st.markdown(f"**👤 Author:** {book['Author']}  ")
                st.markdown(f"**📅 Published:** {book['Published']}  ")
                st.markdown(f"**⭐ Ratings:** {book['Ratings']} | **🏆 Score:** {book['Score']}  ")
                category = extract_book_categories(book['Description'])
                st.markdown(f"**📚 Category:** {category}")
                st.markdown(f"📝 *{book['Description'][:300]}...*")
                st.divider()
