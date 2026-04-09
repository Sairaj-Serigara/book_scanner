import easyocr
import numpy as np
import cv2

reader = easyocr.Reader(['en'])

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text(crops):
    texts = []

    for crop in crops:
        try:
            img = np.array(crop)
            img = preprocess(img)

            result = reader.readtext(img)
            text = " ".join([r[1] for r in result])

            texts.append(text)

        except:
            texts.append("")

    return texts