import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
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

def model_evaluation(y_true, y_pred, y_prob, model_name="Sentiment Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score((y_true, y_pred), y_prob, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(classification_report(y_true, y_pred , target_names=['Negative', 'Neutral', 'Positive'], zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

def analyze_predictions(df, text_col="text", pred_col='predicted_sentiment', true_col=None, confidence_col='prediction_confidence'):
    print("Analyze: \n")
    top_confident = df.nlargest(10, confidence_col)
    for id, row in top_confident.iterrows():
        true_labels = row[true_col] if true_col else "N/A"
        print(f"{row[text_col][:80]}... -> Predict: {row[pred_col]}, "
              f"True: {true_labels}, Confidence: {row[confidence_col]:.4f}")

    low_confident = df.nsmallest(10, confidence_col)
    for idx, row in low_confident.iterrows():
        true_labels = row[true_col] if true_col else "N/A"
        print(f"{row[text_col][:80]}... -> Predict: {row[pred_col]}, "
              f"True: {true_labels}, Confidence: {row[confidence_col]:.4f}")

    pred_distribution = df[pred_col].value_counts().sort_index()
    for class_label, count in pred_distribution.items():
        percentage = (count / len(df)) * 100
        print(f"Class {class_label}: {count} examples ({percentage:.1f}%)")

if "sentiment" in test_data_sample.columns:
    true_labels = test_data_sample["sentiment"]
    metrics = model_evaluation(true_labels, test_pred, test_prob)
    print("\nMy")
    test_accuracy = accuracy_score(test_data_sample["sentiment"], test_pred)
    print("Test Accuracy: ", test_accuracy)
    print(classification_report(test_data_sample["sentiment"], test_pred))
else:
    print(test_data_sample["prediction_confidence"].describe())
    analyze_predictions(test_data_sample)

print("\n")
for index, row in test_data_sample.head(20).iterrows():
    print(f"{row['text']} -> {row['predicted_sentiment']}")