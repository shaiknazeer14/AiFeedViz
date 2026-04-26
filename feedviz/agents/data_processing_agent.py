import os
from dotenv import load_dotenv
load_dotenv("feedviz/.env")

import json
import pandas as pd
from pathlib import Path
from crewai import Agent, Task, LLM
from crewai.tools import tool

from feedviz.tools.text_cleaner import clean_feedback_dataframe, get_processing_summary
from feedviz.config.settings import settings


@tool("Load and validate feedback CSV")
def load_feedback_csv(csv_path: str) -> str:
    """
    Loads the student feedback CSV file from the given path,
    validates that all required columns exist, and returns
    a JSON string summary of the loaded data.
    Use this tool first to load raw feedback data before cleaning.
    """
    try:
        path = Path(csv_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {csv_path}"})
        df = pd.read_csv(path, dtype={
            "student_id": str, "teacher_name": str, "department": str,
            "subject": str, "section": str, "feedback_text": str, "date": str,
        })
        return json.dumps({
            "status": "loaded", "records": len(df),
            "columns": list(df.columns),
            "preview": df.head(3).to_dict(orient="records"),
            "teachers_found": df["teacher_name"].nunique(),
            "departments_found": df["department"].nunique(),
        }, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Clean and preprocess feedback data")
def clean_and_save_feedback(csv_path: str) -> str:
    """
    Loads raw feedback from csv_path, applies full NLP preprocessing
    (lowercase, remove punctuation, stopwords, stemming), adds helper
    columns, and saves the cleaned data to the processed directory.
    Must be called after load_feedback_csv.
    """
    try:
        df_raw = pd.read_csv(csv_path, dtype={
            "student_id": str, "teacher_name": str, "department": str,
            "subject": str, "section": str, "feedback_text": str, "date": str,
        })
        df_clean = clean_feedback_dataframe(df_raw)
        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(settings.cleaned_csv, index=False)
        summary = get_processing_summary(df_clean)
        summary["saved_to"] = str(settings.cleaned_csv)
        summary["new_columns_added"] = [
            "feedback_clean", "feedback_readable",
            "teacher_section_key", "rating_normalized",
        ]
        return json.dumps(summary, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


groq_llm = LLM(
    model="ollama/mistral",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1,
)

data_processing_agent = Agent(
    role="Senior Data Engineer specializing in academic feedback systems",
    goal=(
        "Load student feedback data from CSV files, validate data quality, "
        "apply NLP preprocessing to clean and normalize feedback text, "
        "and produce a reliable, structured dataset for downstream analysis agents."
    ),
    backstory=(
        "You are an experienced data engineer with 8 years of expertise in "
        "building ETL pipelines for educational analytics platforms. "
        "You are meticulous about data quality — you always validate schemas, "
        "handle missing values explicitly, and log every transformation."
    ),
    tools=[load_feedback_csv, clean_and_save_feedback],
    llm=groq_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    use_system_prompt=True,
    respect_context_window=True,
)

data_processing_task = Task(
    description=(
        "Process the student feedback data from the file at: {csv_path}\n\n"
        "Follow these steps in order:\n"
        "1. Load the CSV file and validate all required columns are present\n"
        "2. Apply full NLP preprocessing pipeline\n"
        "3. Handle data quality issues\n"
        "4. Save the cleaned dataset to the processed directory\n"
        "5. Report a full summary of what was processed"
    ),
    expected_output=(
        "A JSON summary containing total_records, teachers, departments, "
        "sections, date_range, avg_rating_overall, feedback_per_teacher, "
        "saved_to, and new_columns_added."
    ),
    agent=data_processing_agent,
    output_file=str(settings.outputs_dir / "data_processing_output.txt"),
)