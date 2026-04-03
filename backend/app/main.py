from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from PIL import Image
import io
import torch

from app.model import load_model
from app.utils import preprocess_image, predict
from app.services.prediction_service import interpret_ripeness
from app.services.intelligence_service import generate_intelligence

# ============================================
# Pydantic Models for API Documentation
# ============================================

class PredictSuccessResponse(BaseModel):
    """Successful prediction response with detailed ripeness analysis"""
    status: str = Field(default="success", description="Operation status")
    ripeness_percent: float = Field(
        ...,
        description="Ripeness percentage (0-100)",
        example=85.5,
        ge=0,
        le=100
    )
    stage: str = Field(
        ...,
        description="Ripeness stage: 'Immature', 'Transitional', or 'Mature'",
        example="Mature"
    )
    confidence: float = Field(
        ...,
        description="Model confidence percentage",
        example=92.5,
        ge=0,
        le=100
    )
    shelf_life: str = Field(
        ...,
        description="Estimated shelf life",
        example="1-2 days"
    )
    quality_score: float = Field(
        ...,
        description="Quality score based on ideal ripeness",
        example=97.5,
        ge=0,
        le=100
    )
    grade: str = Field(
        ...,
        description="Quality grade: 'A+', 'A', 'B+', 'B', 'C', or 'Reject'",
        example="A"
    )
    market: str = Field(
        ...,
        description="Recommended market destination",
        example="Export"
    )
    price_category: str = Field(
        ...,
        description="Price category: 'Premium', 'High', 'Medium-High', 'Medium', 'Low', or 'None'",
        example="High"
    )
    recommendation: str = Field(
        ...,
        description="Harvest or sale recommendation",
        example="Harvest now (ideal)"
    )


class PredictInvalidResponse(BaseModel):
    """Response when input is not recognized as dragon fruit"""
    status: str = Field(default="invalid", description="Operation status - invalid input")
    message: str = Field(
        default="Input not recognized as dragon fruit",
        description="Error message explaining why input is invalid"
    )
    confidence: float = Field(
        ...,
        description="Model confidence (below threshold)",
        example=45.5,
        ge=0,
        le=100
    )


class ErrorResponse(BaseModel):
    """Error response for failed operations"""
    detail: str = Field(..., description="Error message")


# ============================================
# FastAPI App Configuration
# ============================================

app = FastAPI(
    title="Dragon Fruit Ripeness Prediction API",
    description=(
        "AI-powered API for predicting dragon fruit ripeness from images. "
        "This API analyzes uploaded dragon fruit images and provides detailed "
        "ripeness analysis including maturity stage, shelf life estimation, "
        "quality grading, market recommendations, and harvest timing advice."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

device = torch.device("cpu")
model = load_model()


# ============================================
# API Endpoints
# ============================================

@app.get(
    "/",
    summary="API Health Check",
    description="Returns a confirmation message that the API is running",
    responses={
        200: {
            "description": "API is running successfully",
            "content": {
                "application/json": {
                    "example": {"message": "Dragon Fruit Ripeness API Running"}
                }
            }
        }
    }
)
def home():
    """Health check endpoint"""
    return {"message": "Dragon Fruit Ripeness API Running"}


@app.post(
    "/predict",
    summary="Predict Dragon Fruit Ripeness",
    description=(
        "Upload a dragon fruit image to get detailed ripeness analysis. "
        "The API will analyze the image using a deep learning model and return:\n\n"
        "- **Ripeness percentage**: How ripe the fruit is (0-100%)\n"
        "- **Stage**: Categorization as Immature, Transitional, or Mature\n"
        "- **Shelf life**: Estimated remaining shelf life\n"
        "- **Quality score**: Overall quality assessment\n"
        "- **Grade**: Quality grade for market classification\n"
        "- **Market**: Recommended market destination\n"
        "- **Recommendation**: Harvest/sale timing advice\n\n"
        "**Supported image formats**: JPEG, PNG, WEBP\n"
        "**Model confidence threshold**: 70% (inputs below this are rejected)"
    ),
    responses={
        200: {
            "description": "Successful prediction or invalid input",
            "content": {
                "application/json": {
                    "examples": {
                        "success_example": {
                            "summary": "Successful Prediction - Mature Dragon Fruit",
                            "value": {
                                "status": "success",
                                "ripeness_percent": 85.5,
                                "stage": "Mature",
                                "confidence": 92.5,
                                "shelf_life": "1-2 days",
                                "quality_score": 97.5,
                                "grade": "A",
                                "market": "Export",
                                "price_category": "High",
                                "recommendation": "Harvest now (ideal)"
                            }
                        },
                        "immature_example": {
                            "summary": "Immature Dragon Fruit",
                            "value": {
                                "status": "success",
                                "ripeness_percent": 25.3,
                                "stage": "Immature",
                                "confidence": 88.2,
                                "shelf_life": "6-8 days",
                                "quality_score": 40.3,
                                "grade": "C",
                                "market": "Processing Industry",
                                "price_category": "Low",
                                "recommendation": "Not ready for harvest"
                            }
                        },
                        "transitional_example": {
                            "summary": "Transitional Dragon Fruit",
                            "value": {
                                "status": "success",
                                "ripeness_percent": 55.8,
                                "stage": "Transitional",
                                "confidence": 79.4,
                                "shelf_life": "4-6 days",
                                "quality_score": 70.8,
                                "grade": "B+",
                                "market": "Domestic (Tier-1)",
                                "price_category": "Medium-High",
                                "recommendation": "Prepare for harvest in few days"
                            }
                        },
                        "invalid_example": {
                            "summary": "Invalid Input - Not Dragon Fruit",
                            "value": {
                                "status": "invalid",
                                "message": "Input not recognized as dragon fruit",
                                "confidence": 45.5
                            }
                        }
                    }
                }
            }
        },
        400: {
            "description": "Bad Request - Invalid input",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "no_file": {
                            "summary": "No File Uploaded",
                            "value": {"detail": "No file uploaded"}
                        },
                        "invalid_image": {
                            "summary": "Invalid Image Format",
                            "value": {"detail": "Invalid image file"}
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Internal processing error"}
                }
            }
        }
    },
    tags=["Prediction"]
)
async def predict_image(
    file: UploadFile = File(
        ...,
        description=(
            "Dragon fruit image file to analyze. "
            "Supported formats: JPEG, PNG, WEBP. "
            "Recommended: Clear, well-lit image showing the fruit surface."
        ),
        media_type="image/jpeg"
    )
):
    """
    Predict dragon fruit ripeness from uploaded image.
    
    This endpoint accepts an image file and returns comprehensive
    ripeness analysis including quality metrics and market recommendations.
    
    Args:
        file: Image file (JPEG, PNG, WEBP)
        
    Returns:
        dict: Ripeness analysis with quality metrics
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Validate file upload
        if not file or not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file uploaded. Please provide an image file."
            )
        
        # Read and validate image
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty file. Please provide a valid image file."
            )
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid image file. "
                    "Please upload a valid image in JPEG, PNG, or WEBP format."
                )
            )
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            probs = torch.softmax(output, dim=1)
        
        # Interpret results
        result = interpret_ripeness(probs)
        
        # Handle invalid input (low confidence)
        if result["status"] == "invalid":
            return PredictInvalidResponse(**result)
        
        # Generate intelligence insights
        intelligence = generate_intelligence(result["ripeness_percent"])
        
        # Merge results
        final_response = {**result, **intelligence}
        
        return PredictSuccessResponse(**final_response)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================
# Additional Documentation Endpoints
# ============================================

@app.get(
    "/health",
    summary="Detailed Health Check",
    description="Returns detailed API health status including model information",
    tags=["System"],
    responses={
        200: {
            "description": "Detailed health status",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "model_loaded": True,
                        "device": "cpu",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "version": "1.0.0"
    }