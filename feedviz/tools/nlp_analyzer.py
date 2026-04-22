import re
import json
import pandas as pd
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from feedviz.config.settings import settings


# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# -------------------------------
# Sentiment Analysis
# -------------------------------
def analyze_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 1.0,
            "label": "neutral"
        }

    scores = analyzer.polarity_scores(text)

    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    scores["label"] = label
    return scores


# -------------------------------
# Keyword Extraction (Improved)
# -------------------------------
def extract_keywords(text: str, top_n: int = 10) -> list:
    if not isinstance(text, str) or not text.strip():
        return []

    # Normalize text
    text = text.lower()

    # Remove punctuation and tokenize
    tokens = re.findall(r'\b\w+\b', text)

    # Count frequency
    word_counts = Counter(tokens)

    return [word for word, _ in word_counts.most_common(top_n)]


# -------------------------------
# Topic Clustering
# -------------------------------
def get_topic_clusters(keywords: list) -> list:
    topics = {
        "pacing": {"fast", "slow", "rush", "speed", "pace"},
        "clarity": {"clear", "confused", "unclear", "explain", "understand"},
        "engagement": {"boring", "engaging", "interesting", "monotonous", "enthusiastic"},
        "helpfulness": {"help", "support", "approachable", "available"},
        "knowledge": {"knowledge", "expert", "subject", "concept"}
    }

    input_words = set(keywords)

    result = [
        topic
        for topic, topic_words in topics.items()
        if input_words.intersection(topic_words)
    ]

    return result


# -------------------------------
# Full DataFrame Analysis
# -------------------------------
def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Safety check
    required_cols = ["feedback_text", "feedback_clean"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Sentiment
    df["sentiment"] = df["feedback_text"].apply(analyze_sentiment)
    df["sentiment_score"] = df["sentiment"].apply(lambda x: x["compound"])
    df["sentiment_label"] = df["sentiment"].apply(lambda x: x["label"])

    # Keywords + Topics
    df["keywords"] = df["feedback_clean"].apply(extract_keywords)
    df["topics"] = df["keywords"].apply(get_topic_clusters)

    return df


# -------------------------------
# NLP Summary
# -------------------------------
def get_nlp_summary(df: pd.DataFrame) -> dict:
    # Sentiment counts
    sentiment_counts = df["sentiment_label"].value_counts().to_dict()

    # Average sentiment per teacher
    avg_sentiment_per_teacher = (
        df.groupby("teacher_name")["sentiment_score"]
        .mean()
        .round(2)
        .to_dict()
    )

    # Top topics
    all_topics = [
        topic
        for topics in df["topics"]
        if isinstance(topics, list)
        for topic in topics
    ]
    top_topics = [t for t, _ in Counter(all_topics).most_common(5)]

    # Top keywords
    all_keywords = [
        kw
        for keywords in df["keywords"]
        if isinstance(keywords, list)
        for kw in keywords
    ]
    top_keywords = [k for k, _ in Counter(all_keywords).most_common(5)]

    return {
        "sentiment_counts": sentiment_counts,
        "avg_sentiment_per_teacher": avg_sentiment_per_teacher,
        "top_topics": top_topics,
        "top_keywords": top_keywords
    }