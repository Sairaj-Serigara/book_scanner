from PIL import Image

def detect_books(image_path):
    image = Image.open(image_path)
    width, height = image.size

    crops = []

    # 🔥 increase splits (important)
    num_splits = 12   # instead of 5
    split_width = width // num_splits

    for i in range(num_splits):
        left = i * split_width
        right = (i + 1) * split_width

        crop = image.crop((left, 0, right, height))
        crops.append(crop)

    return crops