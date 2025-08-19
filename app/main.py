from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .classifier import ToxicityClassifier


class PredictRequest(BaseModel):
  text: str


class BatchPredictRequest(BaseModel):
  texts: list[str]


model = ToxicityClassifier()

app = FastAPI(title='ML Model API', version='1.0')


@app.post('/predict')
def predict(request: PredictRequest):
  """Predict toxicity for a single text."""
  try:
    result = model.predict_text(request.text)
    return {"input": request.text, "prediction": result}
  except Exception as e:
    raise HTTPException(status_code=400, detail=f'Prediction error: {str(e)}')


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest):
  """Predict toxicity for multiple texts."""
  try:
    result = model.predict_texts(request.texts)
    return {"inputs": request.texts, "predictions": result}
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")



@app.get('/health')
def health():
  """Health check endpoint."""
  return {'status': 'ok'}


@app.get('/model-info')
def model_info():
  """Metadata about the trained model."""
  return {
    'version': '1.0',
    'trained_on': '2025-08-18',
    'f1_macro_score': 0.69,
    'f1_micro_score': 0.74,
    
  }
