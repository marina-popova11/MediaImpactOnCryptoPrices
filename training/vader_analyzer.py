import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score["compound"]
    if score["compound"] >= 0.5:
        sentiment = "positive"
    elif score["compound"] <= -0.5:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return pd.Series([compound, sentiment])

analyzer = SentimentIntensityAnalyzer()
data = pd.read_csv("india-news-headlines.csv")
data_sample = data.sample(1000, random_state=42)
data_sample[["compound", "sentiment"]] = data_sample["headline_text"].apply(get_sentiment)
for index, row in data_sample.head(20).iterrows():
    print(f"{row['headline_text']} -> {row['sentiment']} (compound={row['compound']})")
