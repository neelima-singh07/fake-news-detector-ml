import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os


true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")


true_df["label"] = 1  # Real
fake_df["label"] = 0  # Fake


df = pd.concat([true_df, fake_df])
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]


X_train, X_test, y_train, y_test = train_test_split(df["content"], df["label"], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# aplly Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# accuracy 
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))


os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved.")
