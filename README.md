# 📚 AI Bookshelf Scanner

##  Project Overview

The **AI Bookshelf Scanner** is a machine learning-based project that detects book titles from bookshelf images and recommends similar books. It combines **image processing, OCR, and NLP techniques** to create a smart book discovery system.

---


###  1. Image Input

* Users can upload bookshelf images
* Images are expected to have:

  * Clear text (no blur)
  * 8–10 books per image
  * Proper lighting and straight angle

---

###  2. OCR (Text Extraction)

* Extracted text from images using OCR
* Processed full bookshelf images to detect book titles

---

###  3. Text Cleaning

* Removed noise and unwanted characters
* Converted text to lowercase
* Improved readability of OCR output

---

###  4. Dataset Creation

* Created a custom dataset of book titles
* Collected images:

  * Mainly self-captured bookshelf images
  * Some additional images for variation

---

###  5. Book Matching System

* Implemented similarity matching between OCR text and dataset
* Used:

  * TF-IDF vectorization
  * Cosine similarity / fuzzy matching
* Added confidence threshold:

  * If match score is high → return book
  * If low → return **"Unknown Book"**

---

###  6. Hybrid Matching Logic (Without API)

* System checks dataset for best match
* If no strong match:

  * Marks as **Unknown Book**
* Avoids forcing incorrect predictions

---

###  7. Recommendation System

* Suggests similar books based on detected titles
* Uses text similarity techniques

---

###  8. User Interface

* Built using Streamlit
* Features:

  * Image upload
  * Display extracted text
  * Show detected books
  * Show recommendations

---

##  Key Concepts Used

* Optical Character Recognition (OCR)
* Natural Language Processing (NLP)
* Text similarity (TF-IDF, cosine similarity)
* Basic Machine Learning pipeline

---

##  Workflow

1. Upload bookshelf image
2. Extract text using OCR
3. Clean extracted text
4. Match with dataset
5. Apply confidence threshold
6. Return detected books or "Unknown Book"
7. Recommend similar books

---

##  Dataset Strategy

* 70–80% self-captured images (high quality)
* 20–30% additional images for variation
* Focus on clear, readable book spines

---

## ⚠️ Current Limitations

* Works best with clear and well-lit images
* Limited to books present in dataset
* OCR accuracy depends on image quality
* No external API integration (yet)

---

##  Future Improvements

* Add book spine detection (object detection)
* Improve OCR accuracy with preprocessing
* Expand dataset size
* Integrate external APIs for unknown books
* Enhance recommendation system
* Deploy as web/mobile app

---

##  Conclusion

This project demonstrates a practical application of **Machine Learning, OCR, and NLP** to solve a real-world problem—automating book detection and recommendation from images.


