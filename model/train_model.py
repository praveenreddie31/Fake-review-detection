import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🔥 Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Load dataset
data_path = os.path.join("data", "dataset.csv")
df = pd.read_csv(data_path)

df = df[["text_", "label"]].rename(columns={"text_": "text"})
df = df.dropna()

# 🔥 Use FULL dataset (no sampling)
print("Dataset loaded:", len(df))

# Clean text
df["text"] = df["text"].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Split done")

# 🔥 Strong TF-IDF
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 3),
    stop_words="english",
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorization done")

# 🔥 Strong SVM
model = LinearSVC(C=2.0)

model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔥 FULL DATASET SVM Accuracy: {accuracy * 100:.2f}%\n")

print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_review_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n✅ Model saved successfully!")