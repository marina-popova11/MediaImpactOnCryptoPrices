import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report

from download import preprocess_text

test_data = pd.read_csv("twitter_validation.csv")
print(test_data.shape)
print(test_data.columns.tolist())
test_data.columns = [f"col_{i}" for i in range(test_data.shape[1])]
test_data = test_data.rename(columns={
    f"col_{3}": "text",
    f"col_{2}": "sentiment"
})

test_data["cleaned_text"] = test_data["text"].apply(preprocess_text)
print(test_data.columns.tolist())


loaded_model = joblib.load("sentiment_model.pkl")
loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")

X_test = loaded_tfidf.transform(test_data["cleaned_text"])

test_pred = loaded_model.predict(X_test)
test_prob = loaded_model.predict_proba(X_test)

test_data["predicted_sentiment"] = test_pred
test_data["prediction_confidence"] = np.max(test_prob, axis=1)

def model_evaluation(y_true, y_pred, model_name="Sentiment Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(classification_report(y_true, y_pred , target_names=['Irrelevant', 'Negative', 'Neutral', 'Positive'], zero_division=0))

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

if "sentiment" in test_data.columns:
    true_labels = test_data["sentiment"]
    metrics = model_evaluation(true_labels, test_pred)
    analyze_predictions(test_data, true_col="sentiment")
else:
    print(test_data["prediction_confidence"].describe())
    analyze_predictions(test_data)

print("\n")
for index, row in test_data.head(20).iterrows():
    print(f"{row['text']} -> {row['predicted_sentiment']}")