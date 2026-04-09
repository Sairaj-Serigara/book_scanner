from PIL import Image

def detect_books(image_path):
    image = Image.open(image_path)
    width, height = image.size

    crops = []
    num_splits = 5
    split_width = width // num_splits

    for i in range(num_splits):
        crop = image.crop((i * split_width, 0, (i + 1) * split_width, height))
        crops.append(crop)

    return crops