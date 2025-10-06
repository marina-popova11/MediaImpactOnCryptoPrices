import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from download import preprocess_text

test_data = pd.read_csv("india-news-headlines.csv")
print(test_data.shape)

test_data = test_data.rename(columns={
    f"headline_text": "text",
})

test_data_sample = test_data.sample(n=1000, random_state=42)

test_data_sample["cleaned_text"] = test_data_sample["text"].apply(preprocess_text)
print(test_data_sample.columns.tolist())

loaded_model = joblib.load("sentiment_model.pkl")
loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")

X_test = loaded_tfidf.transform(test_data_sample["cleaned_text"])

test_pred = loaded_model.predict(X_test)
test_prob = loaded_model.predict_proba(X_test)

test_data_sample["predicted_sentiment"] = test_pred
test_data_sample["prediction_confidence"] = np.max(test_prob, axis=1)

if "sentiment" in test_data_sample.columns:
    test_accuracy = accuracy_score(test_data_sample["sentiment"], test_pred)
    print("Test Accuracy: ", test_accuracy)
    print(classification_report(test_data_sample["sentiment"], test_pred))
else:
    print(test_data_sample["prediction_confidence"].value_counts())
for index, row in test_data_sample.head(20).iterrows():
    print(f"{row['text']} -> {row['predicted_sentiment']}")