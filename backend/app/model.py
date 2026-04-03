import os
import urllib.request
import torch
import torch.nn as nn
import timm

# -------------------------------
# DEVICE CONFIG
# -------------------------------
device = torch.device("cpu")

# -------------------------------
# MODEL CONFIG
# -------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ripeness_model.pth")

MODEL_URL = "https://huggingface.co/gagan05/deepvision-ripeness-model/resolve/main/ripeness_model.pth"


# -------------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# -------------------------------
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("⬇ Downloading model from Hugging Face...")

        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(" Model downloaded successfully")

        except Exception as e:
            raise RuntimeError(f" Failed to download model: {e}")

    else:
        print(" Model already exists (cached locally)")


# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model():
    download_model()

    model = timm.create_model('efficientnet_b0', pretrained=False)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)

    except Exception as e:
        raise RuntimeError(f" Model loading failed: {e}")

    model.to(device)
    model.eval()

    print(" Model loaded and ready for inference")

    return model