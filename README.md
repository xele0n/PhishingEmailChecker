# Phishing Email Checker

A deep learning-based tool to detect phishing emails using PyTorch and a user-friendly web interface.

Still in active development, developed by a 15yo

## Features
- Deep learning model trained on a large dataset of phishing and legitimate emails
- Web interface for easy email submission
- Support for both text and image (screenshot) input
- Detailed analysis of why an email might be phishing
- Real-time prediction with confidence scores

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR (required for image processing):
- Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Model Details
The model uses a transformer-based architecture trained on the Phishing Email Dataset from Kaggle. It analyzes various features of emails including:
- Text content
- URL patterns
- Email headers
- Sender information
- Language patterns

## Dataset
The model is trained on the "Phishing Email Dataset" from Kaggle, which contains thousands of labeled phishing and legitimate emails. It takes a while to train, I am yet to optimise it.