import os
from dotenv import load_dotenv
load_dotenv("feedviz/.env")

import json
import pandas as pd
from pathlib import Path
from crewai import Agent, Task, LLM
from crewai.tools import tool

from feedviz.tools.nlp_analyzer import analyze_dataframe, get_nlp_summary
from feedviz.config.settings import settings


@tool("Run NLP analysis on cleaned feedback data")
def run_nlp_analysis(csv_path: str) -> str:
    """
    Loads cleaned feedback CSV, runs sentiment analysis,
    extracts keywords and topics, saves analyzed data
    to processed directory and returns NLP summary.
    """
    try:
        df = pd.read_csv(csv_path)
        analyzed_df = analyze_dataframe(df)
        output_path = Path(settings.processed_data_dir) / "feedback_analyzed.csv"
        analyzed_df.to_csv(output_path, index=False)
        summary = get_nlp_summary(analyzed_df)
        summary["saved_to"] = str(output_path)
        return json.dumps(summary, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Get sentiment summary from analyzed feedback")
def get_sentiment_summary(csv_path: str) -> str:
    """
    Loads already analyzed feedback CSV and returns
    a full NLP summary including sentiment counts,
    average sentiment per teacher, top topics and keywords.
    """
    try:
        df = pd.read_csv(csv_path)
        summary = get_nlp_summary(df)
        return json.dumps(summary, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1,
    tool_choice="auto",
)

nlp_analysis_agent = Agent(
    role="NLP Analysis Specialist for Academic Feedback",
    goal=(
        "Perform sentiment analysis, keyword extraction, and topic clustering "
        "on cleaned student feedback data. Generate a comprehensive NLP summary "
        "for downstream agents."
    ),
    backstory=(
        "You are an expert NLP engineer with deep knowledge of sentiment analysis, "
        "keyword extraction, and topic modeling. You understand how to extract "
        "meaningful insights from short informal text like student feedback."
    ),
    tools=[run_nlp_analysis, get_sentiment_summary],
    llm=groq_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    use_system_prompt=True,
    respect_context_window=True,
)

nlp_analysis_task = Task(
    description=(
        "Perform full NLP analysis on the cleaned feedback data at: "
        "{csv_path}\n\n"
        "Follow these steps:\n"
        "1. Load the cleaned feedback CSV\n"
        "2. Run sentiment analysis on all feedback entries\n"
        "3. Extract top keywords per feedback entry\n"
        "4. Detect topic clusters per feedback entry\n"
        "5. Save analyzed data to processed directory\n"
        "6. Return full NLP summary"
    ),
    expected_output=(
        "A JSON summary containing sentiment_counts, "
        "avg_sentiment_per_teacher, top_topics, top_keywords, saved_to."
    ),
    agent=nlp_analysis_agent,
    output_file=str(settings.outputs_dir / "nlp_analysis_output.txt"),
)