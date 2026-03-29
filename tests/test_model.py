import torch

model_path = "ai-engine/models/ripeness_model.pth"

try:
    torch.load(model_path, map_location="cpu")
    print("Model loaded successfully ✅")
except Exception as e:
    print("Error:", e)