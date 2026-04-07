import pandas as pd
from rapidfuzz import process
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model + dataset once
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("data/books.csv")


# -------------------------------
# 🔍 Fuzzy Match OCR → Real Titles
# -------------------------------
def match_titles(ocr_texts):
    matched_titles = []

    for text in ocr_texts:
        match, score, _ = process.extractOne(text, df['title'])

        if score > 60:  # threshold
            matched_titles.append(match)

    return list(set(matched_titles))


# -------------------------------
# 🎯 Recommendation Function
# -------------------------------
def recommend(user_interest, ocr_texts):

    matched_titles = match_titles(ocr_texts)

    # Filter only detected books
    if matched_titles:
        filtered_df = df[df['title'].isin(matched_titles)].copy()
    else:
        filtered_df = df.copy()

    # Create embeddings
    corpus = filtered_df['description'].tolist() + [user_interest]
    embeddings = model.encode(corpus)

    user_vec = embeddings[-1]
    book_vecs = embeddings[:-1]

    scores = cosine_similarity([user_vec], book_vecs)[0]

    filtered_df['score'] = scores

    # Sort results
    filtered_df = filtered_df.sort_values(by="score", ascending=False)

    return filtered_df, matched_titles