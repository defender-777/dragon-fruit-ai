from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = ['Immature', 'Mature']

def preprocess_image(image: Image.Image):
    return val_transform(image)

def predict(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)

        ripeness = probs[0][1].item() * 100
        stage = classes[torch.argmax(probs).item()]

    return ripeness, stage