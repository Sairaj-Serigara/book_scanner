import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_books(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    boxes = results.xyxy[0].cpu().numpy()
    crops = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box

        # COCO class 73 = book
        if int(cls) == 73:
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            crops.append(crop)

    # fallback if nothing detected
    if len(crops) == 0:
        crops.append(img)

    return crops