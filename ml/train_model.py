"""
train_model.py
--------------
PURPOSE : Load the SMS Spam dataset, preprocess text, train a
          Naive Bayes model with TF-IDF features, evaluate it,
          and save the model + vectorizer as pickle files.

HOW TO RUN : python train_model.py
DATASET    : Download "spam.csv" from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
             and place it in the same folder as this file.
"""

import pandas as pd
import numpy as np
import pickle
import re
import string

# --- NLP tools ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Scikit-learn tools ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK data (only needed once)
nltk.download('stopwords')
nltk.download('punkt')

# ─────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────
print("[1] Loading dataset...")

# The CSV has columns: v1 (label: spam/ham), v2 (message text)
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the two useful columns and rename them clearly
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary numbers: spam=1, ham=0
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

print(f"   Total messages: {len(df)}")
print(f"   Spam count   : {df['label_num'].sum()}")
print(f"   Ham count    : {len(df) - df['label_num'].sum()}")

# ─────────────────────────────────────────
# STEP 2: Text Preprocessing Function
# ─────────────────────────────────────────
print("[2] Preprocessing text...")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Clean and normalize a raw email/SMS message.

    Steps:
    1. Convert to lowercase          -> "FREE PRIZE" becomes "free prize"
    2. Remove punctuation            -> "Hello!" becomes "Hello"
    3. Tokenize into words           -> "free prize" becomes ["free", "prize"]
    4. Remove stopwords              -> remove "the", "is", "at" etc.
    5. Stem each word                -> "winning" becomes "win"
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove punctuation and special characters (keep only letters & spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 3: Tokenize (split into individual words)
    tokens = text.split()

    # Steps 4 & 5: Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    # Rejoin tokens into a single clean string
    return ' '.join(tokens)

# Apply preprocessing to every message
df['clean_message'] = df['message'].apply(preprocess)

print("   Sample original  :", df['message'].iloc[0])
print("   Sample processed :", df['clean_message'].iloc[0])

# ─────────────────────────────────────────
# STEP 3: Feature Extraction with TF-IDF
# ─────────────────────────────────────────
print("[3] Applying TF-IDF vectorization...")

"""
TF-IDF (Term Frequency - Inverse Document Frequency):
- TF  = how often a word appears in THIS message
- IDF = how rare that word is across ALL messages
- TF-IDF is high for words that are common in one message but rare overall
  (e.g., "FREE", "WINNER", "CLICK" appear often in spam but rarely in normal texts)
- max_features=5000 means we keep only the top 5000 most meaningful words
"""
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['clean_message'])   # Feature matrix
y = df['label_num']                            # Labels (0 or 1)

# ─────────────────────────────────────────
# STEP 4: Train / Test Split
# ─────────────────────────────────────────
print("[4] Splitting into train/test sets (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # Fixed seed for reproducibility
)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing samples  : {X_test.shape[0]}")

# ─────────────────────────────────────────
# STEP 5: Train Naive Bayes Model
# ─────────────────────────────────────────
print("[5] Training Multinomial Naive Bayes model...")

"""
Naive Bayes for Text Classification:
- Based on Bayes' Theorem: P(spam | words) ∝ P(words | spam) * P(spam)
- "Naive" assumption: each word is independent (not really true, but works well!)
- MultinomialNB is designed for word count / TF-IDF features
- It's fast, simple, and works very well for spam detection
"""
model = MultinomialNB()
model.fit(X_train, y_train)

# ─────────────────────────────────────────
# STEP 6: Evaluate Model
# ─────────────────────────────────────────
print("[6] Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n   ✔ Accuracy : {accuracy * 100:.2f}%")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print("   Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────
# STEP 7: Save Model and Vectorizer
# ─────────────────────────────────────────
print("[7] Saving model and vectorizer...")

# Save the trained model using pickle
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer (MUST use same vectorizer during prediction)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("   ✔ Saved: spam_model.pkl")
print("   ✔ Saved: tfidf_vectorizer.pkl")
print("\n[DONE] Model training complete!")