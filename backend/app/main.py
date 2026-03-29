from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch

from backend.app.model import load_model
from backend.app.utils import preprocess_image, predict

app = FastAPI()

device = torch.device("cpu")
model = load_model()

@app.get("/")
def home():
    return {"message": "Dragon Fruit Ripeness API Running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = preprocess_image(image)
        ripeness, stage = predict(model, image_tensor, device)

        return {
            "ripeness_percent": round(ripeness, 2),
            "stage": stage
        }

    except Exception as e:
        return {"error": str(e)}