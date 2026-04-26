# feedviz/agents/scoring_agent.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv("feedviz/.env")

from crewai import Agent, Task, LLM
from crewai.tools import tool

from feedviz.tools.scorer import compute_teacher_scores, get_scoring_summary
from feedviz.config.settings import settings


groq_llm = LLM(
    model="ollama/mistral",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1,
)


@tool("Compute teacher scores from analyzed feedback")
def compute_scores(csv_path: str) -> str:
    """
    Loads analyzed feedback CSV, computes section-wise
    and overall teacher performance scores using weighted
    formula combining rating, sentiment and consistency.
    Saves results and returns scoring summary.

    Args:
        csv_path: Path to analyzed feedback CSV.

    Returns:
        JSON string with scoring summary.
    """
    try:
        df = pd.read_csv(csv_path)

        section_df, overall_df = compute_teacher_scores(df)

        # Save outputs
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        section_df.to_csv(
            settings.outputs_dir / "section_scores.csv", index=False
        )
        overall_df.to_csv(
            settings.outputs_dir / "overall_scores.csv", index=False
        )

        summary = get_scoring_summary(section_df, overall_df)
        summary["section_scores_saved"] = str(settings.outputs_dir / "section_scores.csv")
        summary["overall_scores_saved"] = str(settings.outputs_dir / "overall_scores.csv")

        return json.dumps(summary, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Get scoring summary from saved scores")
def get_scores_summary(csv_path: str) -> str:
    """
    Loads already computed scores and returns summary.

    Args:
        csv_path: Path to overall scores CSV.

    Returns:
        JSON string with scores summary.
    """
    try:
        overall_df = pd.read_csv(csv_path)
        return json.dumps({
            "top_teacher": overall_df.loc[
                overall_df["overall_score"].idxmax(), "teacher_name"
            ],
            "lowest_teacher": overall_df.loc[
                overall_df["overall_score"].idxmin(), "teacher_name"
            ],
            "scores": overall_df[[
                "teacher_name", "overall_score", "confidence"
            ]].to_dict(orient="records"),
        }, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


scoring_agent = Agent(
    role="Academic Performance Scoring Specialist",
    goal=(
        "Compute fair and accurate performance scores for all teachers "
        "using a weighted formula combining rating, sentiment, and consistency. "
        "Produce section-wise and overall scores for ranking."
    ),
    backstory=(
        "You are an expert in educational analytics and performance measurement. "
        "You design fair scoring systems that combine quantitative ratings with "
        "qualitative sentiment signals to produce accurate teacher evaluations."
    ),
    tools=[compute_scores, get_scores_summary],
    llm=groq_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    use_system_prompt=True,
    respect_context_window=True,
)

scoring_task = Task(
    description=(
        "Compute performance scores from analyzed feedback at: "
        "{analyzed_csv_path}\n\n"
        "Steps:\n"
        "1. Load analyzed feedback CSV\n"
        "2. Compute section-wise scores for each teacher\n"
        "3. Compute overall scores across all sections\n"
        "4. Save section and overall scores to outputs\n"
        "5. Return scoring summary"
    ),
    expected_output=(
        "A JSON summary containing total_teachers, top_teacher, "
        "lowest_teacher, avg_overall_score, section_scores, "
        "overall_scores, and file paths."
    ),
    agent=scoring_agent,
    output_file=str(settings.outputs_dir / "scoring_output.txt"),
)