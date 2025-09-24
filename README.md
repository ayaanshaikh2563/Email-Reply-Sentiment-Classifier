# SvaraAI Reply Classification Pipeline

## Overview

This project implements a complete machine learning pipeline for classifying email replies into three categories:  
- **Positive**: Interested in meeting/demo  
- **Negative**: Not interested/rejection  
- **Neutral**: Non-committal or irrelevant  

The solution includes:  
- Baseline models (Logistic Regression, LightGBM)  
- Transformer fine-tuning (DistilBERT using Hugging Face)  
- FastAPI deployment for serving the model  
- Interactive frontend UI for inference  
- Docker containerization for production-ready deployment  

---

## 📁 Project Structure
SVARAAI_ASSIGNMENT/
├── Models/
│   ├── best_model/
│   ├── data/
│   ├── lgb_model.pkl
│   ├── lr_model.pkl
│   └── tfidf_vectorizer.pkl
├── results/
├── static/
│   └── index.html (recommended for your frontend)
├── SvaraAI_Assignment/
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── share/
├── __pycache__/
├── .gitignore
├── answers.md
├── app.py
├── Dockerfile
├── notebook-part-a.py
├── README.md
├── requirements.txt
├── setup.sh
└── test-api.py

🚀 Quick Start

1. Environment Setup:
python -m venv venv
source venv/bin/activate     # For Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Run ML Pipeline:
python notebook-part-a.py

This runs data cleaning, trains baseline models and fine-tunes DistilBERT, evaluates them, and saves the best model.

3. API Server Setup:
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
OR
python app.py

To start FastAPI backend.

Notes:
Make sure static/ folder with your frontend (index.html) is in the root directory.
CORS middleware enabled for frontend-backend communication.

4. Use Frontend UI:
http://localhost:8000
Open the browser, it displays an interactive app with an animated starry background, a rotating star, input box, classify button, and result display.

5. Test API (Optional)
Health check:
curl http://localhost:8000/health

Single prediction:
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"Looking forward to the demo!\"}"

Expected response:
{
  "label": "positive",
  "confidence": 0.87
}

🐳 Docker Deployment:
Build container:
docker build -t svaraai-classifier .

Run container:
docker run -p 8000:8000 svaraai-classifier

Access API and frontend on http://localhost:8000

⚙️ Additional Notes
CORS Enabled: To allow frontend JS to communicate with backend API.

Lifespan Event (Optional): Replace deprecated @app.on_event("startup") with lifespan handler for model loading.

Requirements:
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
scikit-learn>=1.1.0
lightgbm>=3.3.0
pandas>=1.4.0
numpy>=1.21.0
fastapi>=0.95.0
uvicorn[standard]>=0.18.0
pydantic>=1.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
requests>=2.28.0

🛠 Troubleshooting
Static folder does not exist: Create static/ folder and put index.html there.

Model loading errors: Verify model folder and tokenizer path ./Models/best_model.

CORS errors: Confirm middleware added in app.py.

Run with uvicorn CLI for hot reload:
uvicorn app:app --reload

📊 Project Recap
Dataset cleaning & augmentation

Baseline (LR & LGBM) and Transformer training & evaluation

Best model served via FastAPI

Interactive, animated frontend

Dockerized app for production deployment

Assignment completed by: Ayaan Shaikh
Date: September 24, 2025



