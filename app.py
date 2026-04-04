import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from rapidfuzz import process
import easyocr
from sentence_transformers import SentenceTransformer, util

# ================= LOAD DATA =================

df = pd.read_csv("books.csv")

df["features"] = (
    df["title"] * 3 + " " +
    df["author"] + " " +
    df["genre"] * 2 + " " +
    df["description"]
)

# ================= LOAD MODELS =================

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

model, reader = load_models()

@st.cache_data
def compute_embeddings(data):
    return model.encode(data.tolist(), convert_to_tensor=True)

book_embeddings = compute_embeddings(df["features"])

# ================= FUNCTIONS =================

def is_blurry(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

def clean_text(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if len(line.strip()) > 3]

# 🔥 NEW: Clean OCR noise
def clean_ocr_noise(text):
    words = text.split()
    return " ".join([w for w in words if len(w) > 3])

def is_valid_text(t):
    return len(t) > 3 and any(c.isalpha() for c in t)

def match_books(extracted, df):
    matched = []
    titles = df["title"].str.lower().tolist()

    for line in extracted:
        match = process.extractOne(line.lower(), titles, score_cutoff=50)
        if match:
            matched.append(df.iloc[match[2]]["title"])

    return list(set(matched))

# ================= UI =================

st.set_page_config(page_title="Smart Shelf Finder", layout="centered")

st.title("📚 Smart Shelf Finder (AI Powered)")
st.info("📸 Upload a bookshelf image")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

matched_books = []

# ================= IMAGE PROCESSING =================

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    if is_blurry(image):
        st.error("❌ Image too blurry")
        st.stop()

    with st.spinner("🔍 Processing image..."):

        img_np = np.array(image)

        # 🔥 Crop middle
        h, w, _ = img_np.shape
        img_np = img_np[int(h*0.2):int(h*0.85), :]

        rotations = [
            img_np,
            cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        all_text = []

        for img in rotations:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            adjusted = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
            edges = cv2.Canny(adjusted, 50, 150)

            kernel = np.ones((2,2), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)

            results = reader.readtext(dilated)

            for res in results:
                if res[2] > 0.25 and is_valid_text(res[1]):
                    all_text.append(res[1])

        text = "\n".join(all_text)

        # 🔥 CLEAN OCR TEXT
        text = clean_ocr_noise(text)

    # DEBUG
    st.write("### 🧾 Extracted Text:")
    st.write(text)

    if len(text.strip()) < 10:
        st.warning("⚠️ Not enough readable text. Try better image.")
        st.stop()

    cleaned = clean_text(text)
    matched_books = match_books(cleaned, df)

    if matched_books:
        st.success("✅ Detected Books:")
        for book in matched_books:
            st.write(f"📘 {book}")

# ================= RECOMMENDATION =================

st.write("### 🔍 Find Books Based on Your Interest")

user_interest = st.text_input("Enter interest (e.g., fiction, history, finance)")

if st.button("Find Books"):

    if not user_interest.strip():
        st.warning("⚠️ Enter interest")
        st.stop()

    user_embedding = model.encode(user_interest, convert_to_tensor=True)

    # 🔥 GLOBAL SEARCH (your requested change)
    scores = []

    for i, emb in enumerate(book_embeddings):
        sim = util.cos_sim(user_embedding, emb).item()
        scores.append((df.iloc[i]["title"], sim))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    st.write("### 🎯 Best Matches:")

    for book, score in scores[:5]:
        st.write(f"📘 {book} (match: {round(score, 2)})")