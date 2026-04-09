import pandas as pd
from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def recommend(interest, ocr_texts):
    df = pd.read_csv("data/books.csv")

    scores = []
    matched_titles = []

    for _, row in df.iterrows():
        title = row['title'].lower()
        genre = row['genre'].lower()

        score = 0

        # 🔹 OCR matching
        for text in ocr_texts:
            for word in text.split():
                sim = similarity(word, title)
                if sim > 0.5:
                    score += sim
                    if row['title'] not in matched_titles:
                        matched_titles.append(row['title'])

        # 🔥 FIX 2: Boost interest weight
        if interest:
            score += 2 * similarity(interest, genre)

        scores.append(score)

    df['score'] = scores
    df = df.sort_values(by='score', ascending=False)

    return df, matched_titles