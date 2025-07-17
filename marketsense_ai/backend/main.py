from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from transformers import pipeline

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LeadInput(BaseModel):
    age: float
    income: float
    purchase_frequency: float

class InsightInput(BaseModel):
    question: str

# Initialize lightweight LLM (distilgpt2, open-source, no API key)
llm = pipeline("text-generation", model="distilgpt2")

@app.post("/predict-lead-score")
async def predict_lead_score(input: LeadInput):
    try:
        model = joblib.load("models/lead_scoring_model.pkl")
        data = pd.DataFrame([input.dict()])
        prediction = model.predict_proba(data)[:, 1][0]
        return {"score": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-insight")
async def ask_insight(input: InsightInput):
    try:
        result = llm(input.question, max_length=100, num_return_sequences=1)
        return {"response": result[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/churn")
async def get_churn():
    try:
        file_path = "notebooks/data/churn_analysis_data.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Churn CSV file not found")
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/segmentation")
async def get_segmentation():
    try:
        file_path = "notebooks/data/customers_with_segments.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Segmentation CSV file not found")
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecasting")
async def get_forecasting():
    try:
        file_path = "notebooks/data/marketing_conversions_forecast.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Forecasting CSV file not found")
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))