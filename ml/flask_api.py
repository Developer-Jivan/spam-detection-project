"""
flask_api.py  —  FIXED VERSION
--------------------------------
Changes made:
  1. Replaced manual after_request CORS headers with flask_cors CORS(app)
     so ALL routes (including error responses) return the correct headers.
  2. Added explicit OPTIONS handler so the browser pre-flight request
     (sent before every cross-origin POST) gets a 200 instead of 404.

HOW TO RUN:
    cd ml/
    python flask_api.py

The API will be available at http://localhost:5000
"""

import pickle
import re
import string

from flask import Flask, request, jsonify
from flask_cors import CORS          # pip install flask-cors (already in requirements.txt)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────
app = Flask(__name__)

# FIX 1: Apply CORS to the entire app — this covers ALL routes and ALL
#         HTTP methods, including the OPTIONS pre-flight that browsers send.
CORS(app)

# ─────────────────────────────────────────
# Load model + vectorizer at startup
# ─────────────────────────────────────────
print("[Flask] Loading model...")
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

print("[Flask] Model loaded successfully!")

# ─────────────────────────────────────────
# Text preprocessing  (must match train_model.py exactly)
# ─────────────────────────────────────────
nltk.download('stopwords', quiet=True)
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Clean and normalize input text before prediction."""
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ─────────────────────────────────────────
# POST /classify
# ─────────────────────────────────────────
@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify():
    """
    Accepts JSON : { "message": "Your email text here" }
    Returns JSON : {
        "label"     : "spam" or "ham",
        "is_spam"   : true / false,
        "spam_prob" : 97.5,   <- percentage
        "ham_prob"  : 2.5
    }
    """
    # FIX 2: Respond immediately to browser pre-flight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json(silent=True)

    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request body'}), 400

    raw_text = data['message']

    if not raw_text.strip():
        return jsonify({'error': 'Empty message provided'}), 400

    # Preprocess → vectorize → predict
    clean_text    = preprocess(raw_text)
    features      = tfidf.transform([clean_text])
    prediction    = model.predict(features)[0]          # 0 = ham, 1 = spam
    probabilities = model.predict_proba(features)[0]

    ham_prob  = round(float(probabilities[0]) * 100, 2)
    spam_prob = round(float(probabilities[1]) * 100, 2)
    label     = 'spam' if prediction == 1 else 'ham'

    print(f"[Predict] '{raw_text[:60]}' => {label.upper()}  (spam: {spam_prob}%)")

    return jsonify({
        'label'    : label,
        'is_spam'  : bool(prediction == 1),
        'spam_prob': spam_prob,
        'ham_prob' : ham_prob,
        'message'  : raw_text[:100]
    }), 200


# ─────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'}), 200


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)