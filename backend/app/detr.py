import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

device = torch.device("cpu")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

model.to(device)
model.eval()


def detect_objects(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    results = processor.post_process_object_detection(
        outputs, threshold=0.8, target_sizes=target_sizes
    )[0]

    detections = []

    for score, _, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        detections.append({
            "confidence": float(score.item()),
            "box": [round(i, 2) for i in box.tolist()]
        })

    return detections


def crop_main_object(image, detections):
    if len(detections) == 0:
        return None

    best = max(detections, key=lambda x: x["confidence"])

    if best["confidence"] < 0.85:
        return None

    x1, y1, x2, y2 = map(int, best["box"])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))