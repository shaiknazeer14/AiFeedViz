import os

from dotenv import load_dotenv
from feedviz.config.settings import settings
import json
import pandas as pd
from crewai import Agent,Task, LLM
from crewai.tools import tool
from feedviz.tools.nlp_analyzer import analyze_sentiment,extract_keywords,get_topic_clusters,analyze_dataframe,get_nlp_summary
load_dotenv()
from pathlib import Path

@tool
def run_nlp_analysis(csv_path)->str:
    df = pd.read_csv(csv_path)
    analyzed_df=analyze_dataframe(df)
    output_path= Path(settings.processed_data_dir) / "feedback_analyzed.csv"
    analyzed_df.to_csv(output_path)

    summary = get_nlp_summary(analyzed_df)
    return json.dumps(summary)

@tool
def get_sentiment_summary(csv_path)->str:
    df = pd.read_csv(csv_path)
    return json.dumps(get_nlp_summary(df))