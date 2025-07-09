import pandas as pd
import numpy as np
import re
import os
import pickle
import string
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# 1. Load and Label Data
# -----------------------------
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 1  # Real
fake_df["label"] = 0  # Fake
df = pd.concat([true_df, fake_df])
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# -----------------------------
# 2. Clean the Text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)      # remove numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["content"] = df["content"].apply(clean_text)

# -----------------------------
# 3. Split the Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# -----------------------------
# 4. Vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5. Model Training
# -----------------------------

# Try Logistic Regression with tuning
log_reg = LogisticRegression(max_iter=1000)
param_grid = {"C": [0.01, 0.1, 1, 10]}
grid = GridSearchCV(log_reg, param_grid, cv=5)
grid.fit(X_train_vec, y_train)
log_model = grid.best_estimator_

# Also try Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
def evaluate_model(name, model):
    print(f"\n📊 Evaluation for: {name}")
    y_pred = model.predict(X_test_vec)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("📋 Classification Report:\n", classification_report(y_test, y_pred))
    print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

evaluate_model("Logistic Regression", log_model)
evaluate_model("Multinomial Naive Bayes", nb_model)

# -----------------------------
# 7. Save the Best Model (choose one)
# -----------------------------
best_model = log_model  # or nb_model if it performed better
os.makedirs("model", exist_ok=True)
pickle.dump(best_model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
print("\n✅ Model and vectorizer saved successfully.")
