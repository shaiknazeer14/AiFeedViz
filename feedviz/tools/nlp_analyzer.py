import json
import pandas as pd
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from feedviz.config.settings import settings

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "label": "neutral"}

    scores = analyzer.polarity_scores(text)

    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    scores["label"] = label
    return scores


# Keyword extraction
def extract_keywords(text:str,top_n:int=10)->list:
    if not isinstance(text, str) or not text.strip():
        return []
    tokens=text.split()
    word_counts=Counter(tokens)
    top_keywords = [word for word, count in word_counts.most_common(top_n)]
    return top_keywords
# Returns list of topics whose keywords overlap with the input keywords
def get_topic_clusters(keywords:list) -> list:
    topics = {
        "pacing": {"fast", "slow", "rush", "speed", "pace"},
        "clarity": {"clear", "confus", "unclear", "explain", "understand"},
        "engagement": {"boring", "engag", "interest", "monoton", "enthusiast"},
        "helpfulness": {"help", "support", "approach", "availabl"},
        "knowledge": {"knowledg", "expert", "subject", "concept"}
    }
    input_words = set(keywords)  # ← use the actual input!
    result = {
        topic: input_words.intersection(topic_words)
        for topic, topic_words in topics.items()
        if input_words.intersection(topic_words)
    }
    return list(result.keys())

# It is to analyze the entire input dataset given and done all functions like analyze_sentiment, extract_keywords etc
def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sentiment"] = df["feedback_text"].apply(analyze_sentiment)

    df["sentiment_score"] = df["sentiment"].apply(lambda x: x["compound"])
    df["sentiment_label"] = df["sentiment"].apply(lambda x: x["label"])

    df["keywords"] = df["feedback_clean"].apply(extract_keywords)
    df["topics"] = df["keywords"].apply(get_topic_clusters)

    return df

def get_nlp_summary(df: pd.DataFrame) -> dict:
    # Step 2 — Count sentiment labels
    sentiment_counts = df['sentiment_label'].value_counts().to_dict()

    # Step 3 — Average sentiment per teacher
    avg_sentiment_per_teacher = (
        df.groupby('teacher_name')['sentiment_score']
        .mean()
        .round(2)
        .to_dict()
    )

    # Step 4 — Most common topics
    all_topics = []
    for topics in df['topics']:
        if isinstance(topics, list):  # ensure it's a list
            all_topics.extend(topics)

    top_topics = [topic for topic, _ in Counter(all_topics).most_common(5)]

    # Step 5 — Most common keywords
    all_keywords = []
    for keywords in df['keywords']:
        if isinstance(keywords, list):  # ensure it's a list
            all_keywords.extend(keywords)

    top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(5)]

    # Step 6 — Return dictionary
    return {
        "sentiment_counts": sentiment_counts,
        "avg_sentiment_per_teacher": avg_sentiment_per_teacher,
        "top_topics": top_topics,
        "top_keywords": top_keywords
    }
