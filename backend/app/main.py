from unittest import result
import os

from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from PIL import Image
import io
import torch

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.model import load_model
from app.utils import preprocess_image, predict
from app.services.prediction_service import interpret_ripeness
from app.services.intelligence_service import generate_intelligence

API_KEY = os.getenv("API_KEY", "default-secret-key")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

device = torch.device("cpu")
model = load_model()

@app.get("/")
@limiter.limit("5/minute")
def home(request: Request):
    return {"message": "Dragon Fruit Ripeness API Running"}

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_image(request: Request, file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
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

        # 🔥 HANDLE INVALID INPUT
        if result["status"] == "invalid":
            return result
        # 🔥 ADD INTELLIGENCE LAYER
        intelligence = generate_intelligence(result["ripeness_percent"])

       # 🔥 MERGE OUTPUT
        final_response = {**result, **intelligence}
         

        return final_response

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))