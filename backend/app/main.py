from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from app.model import load_model, run_inference, validate_dragon_fruit
from app.utils import preprocess_image
from app.services.prediction_service import interpret_ripeness
from app.services.intelligence_service import generate_intelligence
from app.detr import detect_objects, crop_main_object

app = FastAPI()

model = load_model()


# -------------------------------
# USER GUIDANCE (🔥 UX UPGRADE)
# -------------------------------
def guidance_message():
    return "Ensure a single dragon fruit is centered, well-lit, and occupies most of the frame. Avoid cluttered backgrounds."


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/")
def home():
    return {"message": "Dragon Fruit Intelligence API 🚀"}


# -------------------------------
# BASIC PREDICT
# -------------------------------
@app.post("/internal/predict", include_in_schema=False)
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = preprocess_image(image)

        probs = run_inference(model, image_tensor)

        result = interpret_ripeness(probs)

        if result["status"] == "invalid":
            return {
                **result,
                "guidance": guidance_message()
            }

        intelligence = generate_intelligence(result["ripeness_percent"])

        return {**result, **intelligence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# DETECTION ONLY
# -------------------------------
@app.post("/internal/detect", include_in_schema=False)
async def detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        detections = detect_objects(image)

        if len(detections) == 0:
            return {
                "status": "error",
                "message": "No object detected",
                "guidance": guidance_message()
            }

        return {
            "status": "success",
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# SMART PIPELINE (🔥🔥 CORE)
# -------------------------------
@app.post("/predict")
async def predict_smart(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ---------------- DETR ----------------
        detections = detect_objects(image)

        # 🔥 FALLBACK (IMPORTANT FIX)
        if len(detections) == 0:
            cropped = image
        else:
            cropped = crop_main_object(image, detections)

        if cropped is None:
            return {
                "status": "error",
                "message": "Unable to isolate object",
                "guidance": guidance_message()
            }

        # ---------------- MODEL ----------------
        image_tensor = preprocess_image(cropped)

        probs = run_inference(model, image_tensor)

        # 🔥 VALIDATION
        is_valid, confidence = validate_dragon_fruit(probs)

        if not is_valid:
            return {
                "status": "error",
                "message": "Not a Dragon Fruit",
                "confidence": confidence,
                "guidance": guidance_message()
            }

        # 🔥 EDGE CASE FILTER
        if 0.85 < confidence < 0.92:
            return {
                "status": "error",
                "message": "Uncertain prediction",
                "confidence": confidence,
                "guidance": guidance_message()
            }

        result = interpret_ripeness(probs)

        intelligence = generate_intelligence(result["ripeness_percent"])

        return {
            **result,
            **intelligence,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))