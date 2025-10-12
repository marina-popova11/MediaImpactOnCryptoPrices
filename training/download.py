import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

train_data = pd.read_csv("twitter_training.csv", header=None)
train_data.columns = [f"col_{i}" for i in range(train_data.shape[1])]
train_data = train_data.rename(columns={
    f"col_{3}": "text",
    f"col_{2}": "sentiment"
})

print(train_data["sentiment"].value_counts())

new_train_data = pd.read_csv("train_with_sentiment.csv")
new_train_data = new_train_data.rename(columns={"tweet": "text"})
combined_data = pd.concat([train_data[["text", "sentiment"]], new_train_data[["text", "sentiment"]]], ignore_index=True)

print(combined_data.shape)
print(combined_data["sentiment"].value_counts())

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def preprocess_text(text):
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)

    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    text = text.lower().strip()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

combined_data["cleaned_text"] = combined_data["text"].apply(preprocess_text)
print(combined_data.columns.tolist())

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train = tfidf.fit_transform(combined_data["cleaned_text"])
y_train = combined_data["sentiment"]

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train, y_train)
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(classification_report, train_predictions)

joblib.dump(model, 'sentiment_model_1.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer_1.pkl')
print("save")