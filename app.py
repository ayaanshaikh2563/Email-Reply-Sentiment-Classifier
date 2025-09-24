from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os


# Define the FastAPI app first
app = FastAPI(
    title="SvaraAI Reply Classification API",
    description="Classify email replies as positive, negative, or neutral",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev, restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory AFTER app is declared
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve frontend file from static folder on root GET request
@app.get("/")
def read_index():
    file_path = os.path.join("static", "index.html")
    return FileResponse(file_path)


model = None
tokenizer = None
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}


class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    label: str
    confidence: float


@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_path = './Models/best_model'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        print("DistilBERT model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")


def classify_text(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        predicted_label = label_mapping[predicted_class]
    return PredictionOutput(label=predicted_label, confidence=confidence)


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_text: TextInput):
    if not input_text.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    try:
        return classify_text(input_text.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
