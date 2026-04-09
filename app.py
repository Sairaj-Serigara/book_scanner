import streamlit as st
from detector import detect_books
from ocr import extract_text
from recommender import recommend
from PIL import Image
import pandas as pd

# -------------------------------
# 🎯 App Title
# -------------------------------
st.title("📚 AI Bookshelf Scanner")

# -------------------------------
# 🧹 Clean OCR Text
# -------------------------------
def clean_ocr_text(texts):
    cleaned = []
    for t in texts:
        t = t.lower()
        t = ''.join(ch for ch in t if ch.isalnum() or ch.isspace())
        if len(t.strip()) > 3:
            cleaned.append(t.strip())
    return cleaned

# -------------------------------
# 🚫 Remove Garbage OCR
# -------------------------------
def remove_noise(texts):
    clean = []
    for t in texts:
        # keep meaningful text only
        if len(t.split()) >= 2 and any(c.isalpha() for c in t):
            clean.append(t)
    return clean

# -------------------------------
# 📤 Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload bookshelf image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    temp_path = "temp.jpg"
    image.save(temp_path)

    # -------------------------------
    # 🔍 Detect Books
    # -------------------------------
    crops = detect_books(temp_path)

    # -------------------------------
    # 🧾 OCR
    # -------------------------------
    texts = extract_text(crops)
    texts = clean_ocr_text(texts)
    texts = remove_noise(texts)   # ✅ FIX 1 APPLIED

    st.subheader("📚 Extracted Text (OCR)")
    for t in texts:
        st.write(f"📄 {t}")

    # -------------------------------
    # 🔎 Interest Input
    # -------------------------------
    interest = st.text_input("🔍 Enter your interest")

    if interest:
        interest = interest.lower()

        results, matched_titles = recommend(interest, texts)

        # -------------------------------
        # 📘 Detected Books
        # -------------------------------
        st.subheader("📚 Detected Books")
        if matched_titles:
            for title in matched_titles:
                st.write(f"📘 {title}")
        else:
            st.warning("⚠️ No confident book titles detected")

        # -------------------------------
        # 🎯 Recommendations
        # -------------------------------
        st.subheader("🎯 Recommendations")

        strong = results[results['score'] > 0.3]

        if not strong.empty:
            for _, row in strong.iterrows():
                st.success(
                    f"📘 {row['title']} ({row['genre']}) - {round(row['score'],2)}"
                )
        else:
            st.warning("⚠️ Few strong matches found")

        # -------------------------------
        # 📚 You may also like
        # -------------------------------
        st.subheader("📚 You may also like")

        shown = set(strong['title'])
        count = 0

        for _, row in results.iterrows():
            if row['title'] not in shown:
                st.write(
                    f"📘 {row['title']} ({row['genre']}) - {round(row['score'],2)}"
                )
                count += 1
            if count == 3:
                break

        # -------------------------------
        # 🌍 Outside Shelf
        # -------------------------------
        if strong.empty:
            st.subheader("🌍 Outside Shelf Recommendations")

            df = pd.read_csv("data/books.csv")
            outside = df[df['genre'].str.contains(interest, case=False, na=False)]

            if not outside.empty:
                for _, row in outside.head(3).iterrows():
                    st.success(f"📘 {row['title']} ({row['genre']})")
            else:
                st.info("No extra recommendations found")

    else:
        st.info("👆 Enter your interest to get recommendations")