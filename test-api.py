# ============================================================
# SvaraAI Reply Classification API - Testing Script
# ============================================================
# This script tests the FastAPI backend (app.py) by:
# 1. Running a health check on the /health endpoint
# 2. Sending sample test cases to the /predict endpoint
# 3. Comparing the predicted labels with expected labels
# ============================================================

import requests
import json

# ============================================================
# Function: test_api
# - Defines base API URL
# - Contains test cases with expected outputs
# - Runs health check before testing predictions
# ============================================================

def test_api():
    base_url = "http://localhost:8000"
    
    test_cases = [
        {"text": "Looking forward to the demo!", "expected": "positive"},
        {"text": "Not interested, please remove me.", "expected": "negative"},
        {"text": "Can you send more information?", "expected": "neutral"},
        {"text": "This sounds perfect! Let's schedule a meeting.", "expected": "positive"},
        {"text": "Thanks but no thanks.", "expected": "negative"}
    ]
    
    print("🧪 Testing SvaraAI Reply Classification API...")
    print("-" * 50)
    
    # ========================================================
    # Step 1: Health Check
    # Ping the API's /health endpoint to confirm server is live
    # ========================================================
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Model loaded: {health_data.get('model_loaded', False)}")
        else:
            print("❌ Health check failed")
            return
    except requests.exceptions.RequestException:
        print("❌ API is not running. Start the server first with: python app.py")
        return
    
    # ========================================================
    # Step 2: Run Predictions
    # For each test case, call the /predict endpoint
    # and print predicted label, confidence, and expected label
    # ========================================================

    print("\n🎯 Testing predictions...")
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"text": test_case["text"]},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Test {i}: ✅")
                print(f"  Input: {test_case['text']}")
                print(f"  Predicted: {result['label']} (confidence: {result['confidence']:.3f})")
                print(f"  Expected: {test_case['expected']}")
                print()
            else:
                print(f"Test {i}: ❌ Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Test {i}: ❌ Request failed: {e}")
    
    print("🎉 Testing complete!")
    print("💡 Visit http://localhost:8000/docs for interactive API docs")

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    test_api()
