"""FastAPI service for Traffy ticket late prediction."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Traffy Late Prediction API",
    description="Predict whether a Bangkok Traffy ticket will be resolved late (>7 days)",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "traffy_rf_model.joblib"

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained model when API starts."""
    global model
    try:
        # By this point, wait_for_model.sh should have ensured model exists
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH} - this should not happen if wait script worked")
            return
            
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API starting without model - predictions will fail")


class TicketFeatures(BaseModel):
    """Input features for prediction."""
    type: str = Field(..., description="Ticket type (e.g., ปัญหาสาธารณสุข, ปัญหาถนน)")
    organization: str = Field(default="Unknown", description="Organization handling the ticket")
    district: str = Field(..., description="District in Bangkok (e.g., บางรัก)")
    lat: float = Field(..., ge=13.0, le=14.0, description="Latitude (13.0-14.0)")
    lon: float = Field(..., ge=100.0, le=101.0, description="Longitude (100.0-101.0)")
    star: int = Field(default=0, ge=0, le=5, description="Star rating (0-5)")
    count_reopen: int = Field(default=0, ge=0, description="Number of times reopened")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    year: int = Field(..., ge=2020, le=2030, description="Year")
    num_hospitals_in_district: int = Field(default=5, ge=0, description="Number of hospitals in district")
    rain_mm: float = Field(default=0.0, ge=0, description="Rainfall in mm")
    is_rainy_hour: int = Field(default=0, ge=0, le=1, description="1 if rainy hour, 0 otherwise")
    rain_last_3h: float = Field(default=0.0, ge=0, description="Rainfall in last 3 hours (mm)")
    temperature: float = Field(default=30.0, ge=15, le=45, description="Temperature in °C")
    high_temperature: int = Field(default=0, ge=0, le=1, description="1 if temp > 33°C, 0 otherwise")
    wind_speed: float = Field(default=2.0, ge=0, le=30, description="Wind speed in m/s")

    class Config:
        schema_extra = {
            "example": {
                "type": "ปัญหาถนน",
                "organization": "กทม.",
                "district": "บางรัก",
                "lat": 13.75,
                "lon": 100.50,
                "star": 3,
                "count_reopen": 0,
                "hour": 14,
                "dayofweek": 2,
                "month": 6,
                "year": 2023,
                "num_hospitals_in_district": 10,
                "rain_mm": 0.5,
                "is_rainy_hour": 1,
                "rain_last_3h": 1.5,
                "temperature": 32.0,
                "high_temperature": 0,
                "wind_speed": 3.5
            }
        }


class PredictionResponse(BaseModel):
    """Prediction result."""
    prediction: str = Field(..., description="LATE or ON-TIME")
    probability_late: float = Field(..., description="Probability of being late (0-1)")
    is_late: int = Field(..., description="1 if late, 0 if on-time")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    tickets: List[TicketFeatures]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Traffy Late Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "message": "Ready for predictions" if model_loaded else "Waiting for model to be trained"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(ticket: TicketFeatures):
    """
    Predict if a ticket will be resolved late.
    
    Returns:
        - prediction: "LATE" or "ON-TIME"
        - probability_late: Probability of being late (0-1)
        - is_late: 1 if late, 0 if on-time
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first by triggering the Airflow DAG at http://localhost:8080"
        )
    
    try:
        # Convert to DataFrame
        input_dict = ticket.dict()
        df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]
        
        result = "LATE" if prediction == 1 else "ON-TIME"
        
        return PredictionResponse(
            prediction=result,
            probability_late=float(probability),
            is_late=int(prediction)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict for multiple tickets at once.
    
    Returns list of predictions with probabilities.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        tickets_data = [ticket.dict() for ticket in request.tickets]
        df = pd.DataFrame(tickets_data)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": "LATE" if pred == 1 else "ON-TIME",
                "probability_late": float(prob),
                "is_late": int(pred)
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract model details
        info = {
            "model_type": type(model.named_steps['model']).__name__,
            "model_path": str(MODEL_PATH),
            "features_expected": len(model.named_steps['preprocessor'].transformers),
        }
        
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
