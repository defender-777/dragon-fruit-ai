from unittest import result

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from fastapi import HTTPException

from app.model import load_model
from app.utils import preprocess_image, predict
from app.services.prediction_service import interpret_ripeness
from app.services.intelligence_service import generate_intelligence

app = FastAPI()

device = torch.device("cpu")
model = load_model()

@app.get("/")
def home():
    return {"message": "Dragon Fruit Ripeness API Running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        image_bytes = await file.read()

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            raise HTTPException(status_code=400, detail="Invalid image file")

        image_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            probs = torch.softmax(output, dim=1)

        result = interpret_ripeness(probs)

        # HANDLE INVALID INPUT
        if result["status"] == "invalid":
            return result
        # ADD INTELLIGENCE LAYER
        intelligence = generate_intelligence(result["ripeness_percent"])

       # MERGE OUTPUT
        final_response = {**result, **intelligence}
         

        return final_response

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))