from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
import pytesseract
from PIL import Image
import io
import os
from train_model import PhishingEmailClassifier

app = Flask(__name__)

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PhishingEmailClassifier()
model.load_state_dict(torch.load('phishing_model.pth', map_location=device))
model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def analyze_text(text):
    # Tokenize and prepare input
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        phishing_prob = probabilities[0][1].item()
    
    # Generate analysis
    analysis = []
    if phishing_prob > 0.7:
        analysis.append("High probability of phishing due to suspicious content")
    elif phishing_prob > 0.4:
        analysis.append("Moderate risk of phishing - exercise caution")
    else:
        analysis.append("Low probability of phishing")
    
    # Add specific analysis points
    if "urgent" in text.lower() or "immediate" in text.lower():
        analysis.append("Contains urgent language - common in phishing attempts")
    if "click here" in text.lower() or "click below" in text.lower():
        analysis.append("Contains suspicious call-to-action phrases")
    if "password" in text.lower() or "account" in text.lower():
        analysis.append("Requests sensitive information - potential red flag")
    
    return {
        'phishing_probability': phishing_prob,
        'analysis': analysis
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' in request.files:
        # Handle image upload
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        text = pytesseract.image_to_string(image)
    else:
        # Handle text input
        text = request.form.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'})
    
    result = analyze_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 