import torch
import torch.nn as nn
import timm

device = torch.device("cpu")  # backend usually CPU

def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    model.load_state_dict(torch.load("ai-engine/models/ripeness_model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model