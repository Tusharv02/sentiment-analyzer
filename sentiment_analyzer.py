# ============================================================
# MOVIE REVIEW SENTIMENT ANALYZER
# Amazon ML Summer School Portfolio Project
# Author: [Tushar Verma] | MNIT Jaipur
# ============================================================

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  MOVIE SENTIMENT ANALYZER — Amazon MLSS Project")
print("=" * 55)

# ============================================================
# STEP 2: Load & Prepare Data
# Using a real-world text classification dataset
# (Similar to IMDB reviews — two categories: pos / neg)
# ============================================================

print("\n[1/5] Loading dataset...")

# We use 20newsgroups filtered to 2 categories as a proxy
# for binary sentiment (positive vs negative tone)
categories = ['rec.sport.baseball', 'talk.politics.misc']
data = fetch_20newsgroups(subset='all',
                          categories=categories,
                          remove=('headers', 'footers', 'quotes'))

texts = data.data
labels = data.target  # 0 = rec.sport.baseball, 1 = talk.politics.misc

print(f"  Total samples loaded : {len(texts)}")
dist = dict(zip(['Class-0','Class-1'], [labels.tolist().count(0), labels.tolist().count(1)]))
print(f"  Label distribution   : {dist}")

# ============================================================
# STEP 3: Text Preprocessing with TF-IDF
# Converts raw text into numerical features the model can use
# ============================================================

print("\n[2/5] Converting text to numbers (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=5000,    # Use top 5000 most important words
    stop_words='english', # Remove common words like 'the', 'is'
    ngram_range=(1, 2)    # Use single words AND word pairs
)

X = vectorizer.fit_transform(texts)
y = labels

print(f"  Feature matrix shape : {X.shape}")
print(f"  (rows=samples, cols=unique word features)")

# ============================================================
# STEP 4: Train-Test Split
# 80% data for training, 20% for testing
# ============================================================

print("\n[3/5] Splitting into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # Fixed seed for reproducibility
    stratify=y           # Keep class balance in both splits
)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# ============================================================
# STEP 5: Train the Model — Logistic Regression
# A classic, interpretable ML algorithm for classification
# ============================================================

print("\n[4/5] Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,       # Max iterations to converge
    C=1.0,               # Regularization strength
    solver='lbfgs'       # Optimization algorithm
)
model.fit(X_train, y_train)

print("  Model training complete!")

# ============================================================
# STEP 6: Evaluate the Model
# ============================================================

print("\n[5/5] Evaluating model performance...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  ★ Accuracy: {accuracy * 100:.2f}%")
print("\n  Detailed Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Class-0', 'Class-1']))

# ============================================================
# STEP 7: Visualizations
# Two plots: Confusion Matrix + Top Features
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sentiment Analyzer — Model Results',
             fontsize=14, fontweight='bold')

# --- Plot 1: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class-0', 'Class-1'],
            yticklabels=['Class-0', 'Class-1'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual Label')
axes[0].set_xlabel('Predicted Label')

# --- Plot 2: Top 15 Most Predictive Words ---
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
top_pos_idx = np.argsort(coef)[-15:]
top_neg_idx = np.argsort(coef)[:15]

top_features = (list(feature_names[top_neg_idx]) +
                list(feature_names[top_pos_idx]))
top_coefs = (list(coef[top_neg_idx]) +
             list(coef[top_pos_idx]))

colors = ['#E24B4A'] * 15 + ['#1D9E75'] * 15
axes[1].barh(top_features, top_coefs, color=colors)
axes[1].set_title('Top 15 Most Predictive Words per Class')
axes[1].set_xlabel('Feature Coefficient')
axes[1].axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Chart saved as 'model_results.png'")

# ============================================================
# STEP 8: Predict on Custom Text (Demo)
# ============================================================

print("\n" + "=" * 55)
print("  CUSTOM PREDICTION DEMO")
print("=" * 55)

sample_texts = [
    "This is a wonderful performance, truly outstanding!",
    "Terrible experience, completely disappointed and angry.",
    "The game was exciting and the players performed well.",
    "This policy is a complete disaster for the country."
]

for text in sample_texts:
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "Class-0 🟢" if pred == 0 else "Class-1 🔴"
    confidence = max(prob) * 100
    print(f"\n  Text      : {text[:55]}...")
    print(f"  Predicted : {label}  (confidence: {confidence:.1f}%)")

print("\n" + "=" * 55)
print("  PROJECT COMPLETE — Ready for GitHub!")
print("=" * 55)
