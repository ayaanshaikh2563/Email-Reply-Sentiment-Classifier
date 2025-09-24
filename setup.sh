#!/bin/bash

# SvaraAI Assignment Quick Setup Script
# Run this script to set up and execute the complete assignment

echo "🚀 SvaraAI Reply Classification Assignment Setup"
echo "================================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎯 Setup complete! Now you can:"
echo ""
echo "1. Run Part A (ML Pipeline):"
echo "   python notebook-part-a.py"
echo ""
echo "2. Run Part B (API Server):"
echo "   python app.py"
echo ""
echo "3. Test the API (in another terminal):"
echo "   python test-api.py"
echo ""
echo "4. View API docs at: http://localhost:8000/docs"
echo ""
echo "⏰ Remember: Assignment deadline is Sept 24, 2025!"
echo "📝 Don't forget to create your GitHub repo and record the video!"