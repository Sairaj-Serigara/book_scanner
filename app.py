import streamlit as st
from detector import detect_books
from ocr import extract_text
from recommender import recommend
from PIL import Image
import numpy as np

st.title("📚 AI Bookshelf Scanner")

uploaded_file = st.file_uploader("Upload bookshelf image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp file
    temp_path = "temp.jpg"
    image.save(temp_path)

    # Step 1: Detect books
    crops = detect_books(temp_path)

    # Step 2: OCR
    texts = extract_text(crops)

    st.subheader("📚 Extracted Text (OCR)")
    for t in texts:
        st.write(t)

    # Step 3: Interest input
    interest = st.text_input("🔍 Enter your interest")

    if interest:
        results, matched_titles = recommend(interest.lower(), texts)

        st.subheader("📚 Detected Books (Matched)")

        if matched_titles:
            for title in matched_titles:
                st.write(f"📘 {title}")
        else:
            st.warning("⚠️ Could not confidently detect book titles")

        st.subheader("🎯 Recommendations")

        strong = results[results['score'] > 0.5]

        if not strong.empty:
            for _, row in strong.iterrows():
                st.success(f"📘 {row['title']} ({row['genre']}) - Score: {round(row['score'],2)}")
        else:
            st.warning(f"⚠️ No strong match for '{interest}'")

            st.write("📚 Showing closest matches:")
            for _, row in results.head(5).iterrows():
                st.write(f"📘 {row['title']} ({row['genre']}) - Score: {round(row['score'],2)}")