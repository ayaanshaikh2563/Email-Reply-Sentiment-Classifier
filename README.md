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

-----

## Repository Structure

```
SVARAAI_Assignment/
├── Models/              # Saved models and vectorizers
│   ├── best_model/      # Fine-tuned DistilBERT
│   ├── lgb_model.pkl    # LightGBM model
│   ├── lr_model.pkl     # Logistic Regression model
│   └── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── data/                # Dataset(s) used for training/testing
│   └── reply_classification_dataset_1.csv
├── results/             # Optional: evaluation outputs, metrics, plots
├── static/              # Frontend assets
│   └── index.html
├── app.py               # FastAPI backend
├── notebook-part-a.py   # Data preprocessing and model training notebook
├── README.md            # Project overview and instructions
├── requirements.txt     # Python dependencies
├── setup.sh             # Setup/install script
├── test-api.py          # API testing script
└── answers.md           # Assignment answers / notes
```

-----

## 🚀 Quick Start

### 1\. Environment Setup

To set up the project environment, follow these steps:

```bash
python -m venv venv
# For macOS/Linux
source venv/bin/activate
# For Windows
venv\Scripts\activate
pip install -r requirements.txt
```

### 2\. Run ML Pipeline

This script handles data cleaning, trains baseline models, fine-tunes DistilBERT, evaluates them, and saves the best model.

```bash
python notebook-part-a.py
```

### 3\. API Server Setup

Start the FastAPI backend server using one of the following commands:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
# OR
python app.py
```

### 4\. Use Frontend UI

Once the server is running, open your browser and navigate to:

`http://localhost:8000`

This will display an interactive web application with an animated starry background, a rotating star, an input box, a classify button, and a result display.

### 5\. Test API (Optional)

You can test the API endpoints using `curl` or any API client.

**Health check:**

```bash
curl http://localhost:8000/health
```

**Single prediction:**

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"Looking forward to the demo!\"}"
```

**Expected response:**

```json
{
  "label": "positive",
  "confidence": 0.87
}
```

-----

## 🐳 Docker Deployment

To deploy the application using Docker, follow these steps:

**Build container:**

```bash
docker build -t svaraai-classifier .
```

**Run container:**

```bash
docker run -p 8000:8000 svaraai-classifier
```

Access the API and frontend on `http://localhost:8000`.

-----

## ⚙️ Additional Notes

  - **CORS Enabled:** The FastAPI application includes CORS middleware to allow communication between the frontend (served from the same host) and the backend API.
  - **Model Loading:** The model is loaded on application startup for efficient serving. The code uses a lifespan handler, which is the recommended approach for modern FastAPI applications.

-----

## 📊 Project Recap

  - Dataset cleaning & augmentation
  - Baseline (LR & LGBM) and Transformer training & evaluation
  - Best model served via FastAPI
  - Interactive, animated frontend
  - Dockerized app for production deployment

-----

*This assignment was completed by: Ayaan Shaikh* 
-----
*Date: September 24, 2025*
