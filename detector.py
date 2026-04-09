from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_books(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return []

    results = model(image)

    crops = []

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)

            w = x2 - x1
            h = y2 - y1

            # 🔥 allow more objects (not strict)
            if h > w * 1.2:   # less strict than before
                crop = image[y1:y2, x1:x2]

                if crop.size > 0:
                    crops.append(crop)

    return crops