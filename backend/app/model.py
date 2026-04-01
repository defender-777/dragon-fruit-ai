from pathlib import Path
import torch
import torch.nn as nn
import timm

device = torch.device("cpu")

# ✅ Correct Docker-safe path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ripeness_model.pth"


def load_model():
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