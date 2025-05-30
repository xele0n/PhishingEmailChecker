from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
import pytesseract
from PIL import Image
import io
import os
import glob
from train_model import PhishingEmailClassifier
import re
from urllib.parse import urlparse

app = Flask(__name__)

def find_best_model():
    """Find the best available model to load"""
    model_options = [
        ('phishing_model.pth', 'Final trained model'),
        ('best_phishing_model.pth', 'Best validation model'),
        ('checkpoints/latest_checkpoint.pth', 'Latest checkpoint')
    ]
    
    for model_path, description in model_options:
        if os.path.exists(model_path):
            print(f"🔍 Found {description}: {model_path}")
            return model_path, description
    
    # If no models found, look for any checkpoint files
    checkpoint_files = glob.glob('checkpoints/checkpoint_*.pth')
    if checkpoint_files:
        # Sort by modification time (newest first)
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"🔍 Found checkpoint: {latest_checkpoint}")
        return latest_checkpoint, 'Checkpoint file'
    
    raise FileNotFoundError("No trained model found! Please run train_model.py first.")

def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = PhishingEmailClassifier()
    
    # Load model state from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print checkpoint info
    epoch = checkpoint.get('epoch', 'Unknown')
    batch_idx = checkpoint.get('batch_idx', 'Unknown')
    best_val_acc = checkpoint.get('best_val_acc', 'Unknown')
    timestamp = checkpoint.get('timestamp', 'Unknown')
    
    print(f"📊 Checkpoint Info:")
    print(f"   Epoch: {epoch}")
    print(f"   Batch: {batch_idx}")
    print(f"   Best Validation Accuracy: {best_val_acc}")
    print(f"   Saved: {timestamp}")
    
    return model

def load_model():
    """Load the best available model"""
    try:
        model_path, description = find_best_model()
        print(f"🤖 Loading model: {description}")
        
        model = PhishingEmailClassifier()
        
        if 'checkpoint' in model_path:
            # Load from checkpoint
            model = load_model_from_checkpoint(model_path)
        else:
            # Load regular model file
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully from: {model_path}")
        return model, model_path
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

try:
    model, model_path = load_model()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("🎯 Phishing Email Checker is ready!")
except Exception as e:
    print(f"💥 Failed to initialize model: {e}")
    print("Please run 'python train_model.py' to train a model first.")
    model = None
    tokenizer = None

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
    
    # Convert text to lowercase for analysis
    text_lower = text.lower()
    
    # Initialize analysis components
    phishing_indicators = []
    legitimate_indicators = []
    suspicious_elements = []
    
    # 1. URGENCY AND PRESSURE TACTICS
    urgency_words = ['urgent', 'immediate', 'asap', 'expires today', 'act now', 'limited time', 
                     'expires soon', 'final notice', 'last chance', 'time sensitive']
    found_urgency = [word for word in urgency_words if word in text_lower]
    if found_urgency:
        phishing_indicators.append(f"🚨 Creates urgency with phrases: {', '.join(found_urgency)}")
    
    # 2. REQUESTS FOR SENSITIVE INFORMATION
    sensitive_requests = {
        'password': ['password', 'pwd', 'passcode'],
        'personal info': ['ssn', 'social security', 'date of birth', 'mother maiden name'],
        'financial info': ['credit card', 'bank account', 'routing number', 'pin number'],
        'login credentials': ['username', 'user id', 'login', 'sign in credentials']
    }
    
    for category, keywords in sensitive_requests.items():
        found_keywords = [word for word in keywords if word in text_lower]
        if found_keywords:
            phishing_indicators.append(f"🔐 Requests {category}: mentions {', '.join(found_keywords)}")
    
    # 3. SUSPICIOUS LINKS AND ACTIONS
    action_phrases = ['click here', 'click below', 'download now', 'verify now', 'update now',
                     'confirm identity', 'validate account', 'reactivate account']
    found_actions = [phrase for phrase in action_phrases if phrase in text_lower]
    if found_actions:
        phishing_indicators.append(f"⚠️ Suspicious call-to-action: {', '.join(found_actions)}")
    
    # 4. FINANCIAL/ACCOUNT THREATS
    threats = ['suspend', 'suspended', 'freeze', 'frozen', 'closed', 'terminated', 'deactivated',
              'unauthorized access', 'security breach', 'compromised']
    found_threats = [threat for threat in threats if threat in text_lower]
    if found_threats:
        phishing_indicators.append(f"💰 Account threats detected: {', '.join(found_threats)}")
    
    # 5. SUSPICIOUS SENDER PATTERNS
    if 'no-reply' in text_lower or 'noreply' in text_lower:
        suspicious_elements.append("📧 Uses 'no-reply' address (common in phishing)")
    
    # 6. GRAMMATICAL AND SPELLING ISSUES
    grammar_issues = ['recieve', 'seperate', 'occured', 'thier', 'loose', 'you\'re account',
                     'it\'s urgent', 'click hear', 'you have recieved']
    found_grammar = [issue for issue in grammar_issues if issue in text_lower]
    if found_grammar:
        phishing_indicators.append(f"📝 Grammar/spelling errors: {', '.join(found_grammar)}")
    
    # 7. MONEY/REWARD OFFERS
    money_offers = ['won', 'winner', 'lottery', 'prize', 'reward', 'cash', 'refund',
                   'compensation', 'inheritance', 'million dollars']
    found_money = [offer for offer in money_offers if offer in text_lower]
    if found_money:
        phishing_indicators.append(f"💸 Unexpected money offers: {', '.join(found_money)}")
    
    # 8. LEGITIMATE INDICATORS
    legitimate_signs = {
        'professional language': ['sincerely', 'best regards', 'thank you for your business'],
        'specific details': ['order number', 'invoice', 'reference number', 'transaction id'],
        'proper contact info': ['customer service', 'help desk', 'support team']
    }
    
    for category, keywords in legitimate_signs.items():
        found_keywords = [word for word in keywords if word in text_lower]
        if found_keywords:
            legitimate_indicators.append(f"✅ Shows {category}: {', '.join(found_keywords)}")
    
    # 9. URL ANALYSIS (basic pattern detection)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    if urls:
        suspicious_urls = []
        for url in urls:
            if any(suspicious in url.lower() for suspicious in ['bit.ly', 'tinyurl', 'short', 'redirect']):
                suspicious_urls.append(url)
        if suspicious_urls:
            phishing_indicators.append(f"🔗 Contains suspicious shortened URLs")
    
    # 10. IMPERSONATION INDICATORS
    impersonation = ['microsoft', 'google', 'amazon', 'paypal', 'bank of america', 'wells fargo',
                    'chase', 'apple', 'facebook', 'netflix', 'irs', 'fbi']
    found_impersonation = [brand for brand in impersonation if brand in text_lower]
    if found_impersonation:
        suspicious_elements.append(f"🎭 Claims to be from: {', '.join(found_impersonation)} (verify authenticity)")
    
    # GENERATE COMPREHENSIVE ANALYSIS
    analysis = []
    
    # Overall assessment
    if phishing_prob > 0.8:
        analysis.append("🚨 **HIGH RISK**: This email shows strong signs of being a phishing attempt")
    elif phishing_prob > 0.6:
        analysis.append("⚠️ **MEDIUM-HIGH RISK**: This email has several concerning phishing indicators")
    elif phishing_prob > 0.4:
        analysis.append("🟡 **MEDIUM RISK**: This email shows some suspicious characteristics")
    elif phishing_prob > 0.2:
        analysis.append("🟢 **LOW RISK**: This email appears mostly legitimate with minor concerns")
    else:
        analysis.append("✅ **VERY LOW RISK**: This email shows strong signs of being legitimate")
    
    # Add specific indicators found
    if phishing_indicators:
        analysis.append("**🚩 Phishing Indicators Found:**")
        analysis.extend(phishing_indicators)
    
    if suspicious_elements:
        analysis.append("**🔍 Suspicious Elements:**")
        analysis.extend(suspicious_elements)
    
    if legitimate_indicators:
        analysis.append("**✅ Legitimate Indicators:**")
        analysis.extend(legitimate_indicators)
    
    # Recommendations based on risk level
    if phishing_prob > 0.6:
        analysis.append("**🛡️ Recommendations:**")
        analysis.append("• Do NOT click any links or download attachments")
        analysis.append("• Do NOT provide any personal information")
        analysis.append("• Verify sender through official channels")
        analysis.append("• Report as phishing to your IT department")
    elif phishing_prob > 0.3:
        analysis.append("**🛡️ Recommendations:**")
        analysis.append("• Exercise caution with any links or attachments")
        analysis.append("• Verify the sender's identity independently")
        analysis.append("• Check URLs carefully before clicking")
    else:
        analysis.append("**💡 General Security Tips:**")
        analysis.append("• Always verify unexpected requests independently")
        analysis.append("• Check sender email addresses carefully")
        analysis.append("• When in doubt, contact the organization directly")
    
    # AI Model confidence note
    confidence = abs(phishing_prob - 0.5) * 2  # Convert to confidence scale
    analysis.append(f"**🤖 AI Model Confidence:** {confidence:.1%} (Phishing probability: {phishing_prob:.1%})")
    
    return {
        'phishing_probability': phishing_prob,
        'analysis': analysis
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train a model first.'})
        
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

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Endpoint to reload the model (useful during training)"""
    global model, model_path
    
    try:
        model, new_model_path = load_model()
        print(f"🔄 Model reloaded from: {new_model_path}")
        return jsonify({
            'success': True, 
            'message': f'Model reloaded from: {new_model_path}',
            'model_path': new_model_path
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        })

@app.route('/model-info')
def model_info():
    """Get information about the currently loaded model"""
    if model is None:
        return jsonify({'error': 'No model loaded'})
    
    return jsonify({
        'model_path': model_path,
        'device': str(device),
        'model_loaded': True
    })

if __name__ == '__main__':
    if model is not None:
        print("🚀 Starting Flask app...")
        print("💡 Tip: Use /reload-model endpoint to load newer checkpoints during training")
    app.run(debug=True) 