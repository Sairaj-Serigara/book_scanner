import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import cv2
import re
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    bert = SentenceTransformer('all-MiniLM-L6-v2')
    yolo = YOLO("yolov8n.pt")
    return bert, yolo

model, yolo_model = load_models()

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv")
    df['combined'] = df['title'] + " " + df['genre'] + " " + df['description']
    return df

df = load_data()

# -------------------------------
# EMBEDDINGS
# -------------------------------
@st.cache_data
def create_embeddings(texts):
    return model.encode(texts, show_progress_bar=False)

embeddings = create_embeddings(df['combined'].tolist())

# -------------------------------
# OCR
# -------------------------------
reader = easyocr.Reader(['en'], gpu=False)

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# RELAXED VALIDATION
# -------------------------------
def is_valid_text(text):
    return len(text) > 5

# -------------------------------
# YOLO DETECT + CROP
# -------------------------------
def detect_and_crop_books(image):
    img = np.array(image)
    results = yolo_model(img)

    crops = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]

            if crop.shape[0] > 40 and crop.shape[1] > 15:
                crops.append(crop)

    return crops

# -------------------------------
# OCR WITH ROTATION
# -------------------------------
def extract_text_from_crops(crops):
    texts = []

    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        rotations = [
            gray,
            cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        best_text = ""

        for img in rotations:
            _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
            result = reader.readtext(thresh)

            text = " ".join([res[1] for res in result])
            text = clean_text(text)

            if len(text) > len(best_text):
                best_text = text

        if is_valid_text(best_text):
            texts.append(best_text)

    return texts

# -------------------------------
# MATCH BOOKS (RELAXED)
# -------------------------------
def match_books_from_texts(texts):
    detected_books = []

    for text in texts:
        query_embedding = model.encode([text])
        sims = cosine_similarity(query_embedding, embeddings)[0]

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score < 0.30:
            continue

        # Confidence levels
        if best_score > 0.60:
            confidence = "✅ High"
        elif best_score > 0.45:
            confidence = "🟡 Medium"
        else:
            confidence = "⚠️ Low"

        detected_books.append({
            "title": df.iloc[best_idx]['title'],
            "genre": df.iloc[best_idx]['genre'],
            "score": round(float(best_score), 2),
            "description": df.iloc[best_idx]['description'],
            "ocr": text,
            "confidence": confidence
        })

    return detected_books

# -------------------------------
# REMOVE DUPLICATES
# -------------------------------
def remove_duplicates(books):
    seen = set()
    unique = []
    for book in books:
        if book['title'] not in seen:
            seen.add(book['title'])
            unique.append(book)
    return unique

# -------------------------------
# GENRE DETECTION
# -------------------------------
def detect_genre_bert(text):
    genres = ["fiction", "history", "finance", "business", "self-help", "productivity"]

    scores = {}
    for g in genres:
        score = cosine_similarity(
            model.encode([text]),
            model.encode([g])
        )[0][0]
        scores[g] = score

    return max(scores, key=scores.get)

# -------------------------------
# RECOMMENDATION
# -------------------------------
def recommend_books(user_input, genre_filter=None):
    query_embedding = model.encode([user_input])
    sims = cosine_similarity(query_embedding, embeddings)[0]

    df_copy = df.copy()
    df_copy['similarity'] = sims

    if genre_filter:
        df_copy = df_copy[df_copy['genre'].str.lower() == genre_filter.lower()]

    df_copy = df_copy[df_copy['similarity'] > 0.35]

    return df_copy.sort_values(by='similarity', ascending=False).head(5)

# -------------------------------
# UI
# -------------------------------
st.title("📚 AI Bookshelf Scanner (Stable Version)")

uploaded_file = st.file_uploader("📸 Upload Bookshelf Image", type=["jpg","png","jpeg"])

detected_genre = None

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    with st.spinner("🔍 Processing..."):
        crops = detect_and_crop_books(image)
        texts = extract_text_from_crops(crops)
        books = match_books_from_texts(texts)
        books = remove_duplicates(books)

    # 🔥 FALLBACK
    if not books:
        st.warning("⚠️ Using fallback (full image OCR)...")
        texts = extract_text_from_crops([np.array(image)])
        books = match_books_from_texts(texts)

    st.subheader("📚 Detected Books")

    if not books:
        st.error("❌ Still no books detected")
    else:
        for b in books:
            st.markdown(f"📘 **{b['title']}**")
            st.caption(f"Genre: {b['genre']} | Match: {b['score']} | {b['confidence']}")
            st.write(b['description'])
            st.write(f"🧾 OCR: {b['ocr']}")
            st.write("---")

    if texts:
        detected_genre = detect_genre_bert(" ".join(texts))
        st.subheader("🧠 Auto Detected Genre")
        st.success(detected_genre)

# -------------------------------
# USER INPUT
# -------------------------------
user_input = st.text_input("🔍 Enter your interest")

user_genre = st.selectbox(
    "🎯 Select Genre",
    ["", "fiction", "self-help", "finance", "business", "history", "productivity"]
)

# -------------------------------
# RECOMMEND
# -------------------------------
if st.button("🚀 Get Recommendations"):

    if not user_input:
        st.warning("Enter your interest")
    else:
        if detected_genre and user_genre and detected_genre != user_genre:
            st.warning(f"⚠️ Mismatch: Image suggests '{detected_genre}' but you selected '{user_genre}'")

        final_genre = user_genre if user_genre else detected_genre

        recs = recommend_books(user_input, final_genre)

        if len(recs) == 0:
            st.error("No strong matches found")
        else:
            st.subheader("🎯 Recommended Books")

            for _, row in recs.iterrows():
                st.markdown(f"📘 **{row['title']}**")
                st.caption(f"Genre: {row['genre']} | Match: {round(row['similarity'],2)}")
                st.write(row['description'])
                st.write("---")