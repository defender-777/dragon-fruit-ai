import os
import urllib.request
import torch
import torch.nn as nn
import timm

device = torch.device("cpu")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ripeness_model.pth")

MODEL_URL = "https://huggingface.co/gagan05/deepvision-ripeness-model/resolve/main/ripeness_model.pth"


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_model():
    download_model()

    model = timm.create_model('efficientnet_b0', pretrained=False)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def run_inference(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probs = torch.softmax(output, dim=1)

    return probs


# 🔥 STRICT VALIDATION
def validate_dragon_fruit(probs, threshold=0.90):
    confidence, predicted = torch.max(probs, dim=1)
    confidence = confidence.item()

    if confidence < threshold:
        return False, confidence

    return True, confidence