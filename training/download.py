import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# from preprocess import preprocess_text

train_data = pd.read_csv("twitter_training.csv", header=None)
print("Train_data shape: ", train_data.shape)
print(train_data.head())

for i in range(3):
    print(f"row {i}: {train_data.iloc[i].tolist()}")
text_column = 3
sentiment_column = 2
train_data.columns = [f"col_{i}" for i in range(train_data.shape[1])]
train_data = train_data.rename(columns={
    f"col_{text_column}": "text",
    f"col_{sentiment_column}": "sentiment"
})

print(train_data["sentiment"].value_counts())

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

train_data["cleaned_data"] = train_data["text"].apply(preprocess_text)
print(train_data.columns.tolist())

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train = tfidf.fit_transform(train_data["cleaned_data"])
y_train = train_data["sentiment"]

print(f"TF-IDF matrix shape: {X_train.shape}")

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

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("save")