import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import cloudpickle

# Load model and DictVectorizer
with open("deployed_models/model_bundle.pkl", "rb") as f_in: # model path with respect to accessibility with containerization (/app workdir)
   model_bundle = cloudpickle.load(f_in)

model = model_bundle["model"]
print(f"Bike Demand Predictor Model loaded...!")
dv = model_bundle["dv"]
print(f"Dict Vectorizer loaded...!")

# Generic input: accepts any feature names/values
class FeaturesInput(BaseModel):
    features: Dict[str, Any]

app = FastAPI(title="bike-rental-prediction")
print(f"Bike Demand Predictor Service Deployed...!")

@app.get("/")
def read_root():
    return {"message": "Ready to serve you bike rental count prediction"}

@app.post("/predict")
async def predict(input_data: FeaturesInput):
    try:
        # Use features as a dict
        features_dict = input_data.features
        # DictVectorizer expects a list of dicts (even for one row)
        X = dv.transform([features_dict])
        prediction = model.predict(X)[0]
        nologprediction = np.exp(prediction)
        return {"prediction": int(round(nologprediction, 0))}
    except Exception as e:
        return {"error": str(e)}