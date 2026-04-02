from unittest import result
import logging
import time

from fastapi import FastAPI, UploadFile, File, Request
from PIL import Image
import io
import torch
from fastapi import HTTPException

from app.model import load_model
from app.utils import preprocess_image, predict
from app.services.prediction_service import interpret_ripeness
from app.services.intelligence_service import generate_intelligence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_logger")

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Completed request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} - Error: {str(e)} - Time: {process_time:.4f}s")
        raise

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

        # 🔥 HANDLE INVALID INPUT
        if result["status"] == "invalid":
            return result
        # 🔥 ADD INTELLIGENCE LAYER
        intelligence = generate_intelligence(result["ripeness_percent"])

       # 🔥 MERGE OUTPUT
        final_response = {**result, **intelligence}
         
        logger.info(f"Prediction result: {final_response}")

        return final_response

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.exception("Unexpected error during processing")
        raise HTTPException(status_code=500, detail=str(e))