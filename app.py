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
# 📤 Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload bookshelf image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # Save temp image
    temp_path = "temp.jpg"
    image.save(temp_path)

    # -------------------------------
    # 🔍 Step 1: Detect Books
    # -------------------------------
    crops = detect_books(temp_path)

    # -------------------------------
    # 🧾 Step 2: OCR
    # -------------------------------
    texts = extract_text(crops)

    st.subheader("📚 Extracted Text (OCR)")
    for t in texts:
        st.write(t)

    # -------------------------------
    # 🔎 Step 3: User Interest
    # -------------------------------
    interest = st.text_input("🔍 Enter your interest")

    if interest:
        interest = interest.lower()

        results, matched_titles = recommend(interest, texts)

        # -------------------------------
        # 📘 Detected Books
        # -------------------------------
        st.subheader("📚 Detected Books (Matched)")

        if matched_titles:
            for title in matched_titles:
                st.write(f"📘 {title}")
        else:
            st.warning("⚠️ Could not confidently detect book titles")

        # -------------------------------
        # 🎯 Strong Recommendations
        # -------------------------------
        st.subheader("🎯 Recommendations")

        strong = results[results['score'] > 0.5]

        if not strong.empty:
            for _, row in strong.iterrows():
                st.success(
                    f"📘 {row['title']} ({row['genre']}) - Score: {round(row['score'],2)}"
                )
        else:
            st.warning(f"⚠️ Only few strong matches for '{interest}'")

        # -------------------------------
        # 📚 You may also like (NO DUPLICATES)
        # -------------------------------
        st.subheader("📚 You may also like:")

        shown_titles = set(strong['title'])

        count = 0
        for _, row in results.iterrows():
            if row['title'] not in shown_titles:
                st.write(
                    f"📘 {row['title']} ({row['genre']}) - Score: {round(row['score'],2)}"
                )
                count += 1
            if count == 3:
                break

        # -------------------------------
        # 🌍 Outside Shelf Recommendations
        # -------------------------------
        if strong.empty:
            st.subheader("🌍 Recommended outside your shelf:")

            full_df = pd.read_csv("data/books.csv")

            outside = full_df[full_df['genre'] == interest]

            if not outside.empty:
                for _, row in outside.head(3).iterrows():
                    st.success(f"📘 {row['title']} ({row['genre']})")
            else:
                st.info("No additional recommendations found for this interest.")