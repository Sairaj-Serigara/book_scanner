import easyocr

reader = easyocr.Reader(['en'])

def extract_text(images):
    texts = []

    for img in images:
        result = reader.readtext(img, detail=0, paragraph=True)

        # Clean text
        clean_text = " ".join(result).lower()

        texts.append(clean_text)

    return texts