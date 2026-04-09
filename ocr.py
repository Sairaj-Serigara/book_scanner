import easyocr
import numpy as np
import cv2

reader = easyocr.Reader(['en'])

def extract_text(crops):
    texts = []

    for crop in crops:
        try:
            # 🔥 ROTATE IMAGE (IMPORTANT)
            rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

            result = reader.readtext(rotated)

            text = " ".join([r[1] for r in result])
            texts.append(text)

        except:
            texts.append("")

    return texts