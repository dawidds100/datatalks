from fastapi import FastAPI
from pydantic import BaseModel
import pickle

with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

app = FastAPI()

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    data = [lead.dict()]
    proba = pipeline.predict_proba(data)[0, 1]
    return {"conversion_probability": proba}
