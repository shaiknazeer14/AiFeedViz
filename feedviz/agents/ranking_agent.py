# feedviz/agents/ranking_agent.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv("feedviz/.env")

from crewai import Agent, Task, LLM
from crewai.tools import tool

from feedviz.tools.ranker import (
    rank_teachers_by_department,
    rank_teachers_globally,
    get_ranking_summary,
)
from feedviz.config.settings import settings


groq_llm = LLM(
    model="ollama/mistral",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1,
)


@tool("Rank teachers by department and globally")
def rank_teachers(overall_scores_path: str) -> str:
    """
    Loads overall scores CSV, ranks teachers within
    each department and globally across all departments.
    Saves ranked results and returns ranking summary.

    Args:
        overall_scores_path: Path to overall_scores.csv

    Returns:
        JSON string with complete ranking summary.
    """
    try:
        overall_df = pd.read_csv(overall_scores_path)

        # Department ranking
        dept_ranked_df = rank_teachers_by_department(overall_df)

        # Global ranking
        global_ranked_df = rank_teachers_globally(dept_ranked_df)

        # Save rankings
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        global_ranked_df.to_csv(
            settings.outputs_dir / "rankings.csv", index=False
        )

        # Save as JSON
        summary = get_ranking_summary(global_ranked_df)
        with open(settings.rankings_output, "w") as f:
            json.dump(summary, f, indent=2)

        summary["saved_to"] = str(settings.rankings_output)
        return json.dumps(summary, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Get ranking summary from saved rankings")
def get_rankings(rankings_path: str) -> str:
    """
    Loads saved rankings JSON and returns the summary.

    Args:
        rankings_path: Path to rankings.json

    Returns:
        JSON string with rankings summary.
    """
    try:
        with open(rankings_path, "r") as f:
            rankings = json.load(f)
        return json.dumps(rankings, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


ranking_agent = Agent(
    role="Academic Performance Ranking Specialist",
    goal=(
        "Rank teachers fairly within each department and globally "
        "based on their overall performance scores. Handle edge cases "
        "like low confidence teachers and single section teachers."
    ),
    backstory=(
        "You are an expert in academic performance evaluation and ranking systems. "
        "You ensure fair and transparent rankings that account for data confidence, "
        "department context, and cross-section consistency."
    ),
    tools=[rank_teachers, get_rankings],
    llm=groq_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    use_system_prompt=True,
    respect_context_window=True,
)

ranking_task = Task(
    description=(
        "Rank all teachers using overall scores from: "
        f"{settings.outputs_dir / 'overall_scores.csv'}\n\n"
        "Steps:\n"
        "1. Load overall scores CSV\n"
        "2. Rank teachers within each department\n"
        "3. Rank teachers globally across all departments\n"
        "4. Handle low confidence teachers\n"
        "5. Save rankings and return summary"
    ),
    expected_output=(
        "A JSON summary containing global_top_3, global_bottom_3, "
        "department_leaderboard, global_leaderboard, "
        "and low_confidence_teachers."
    ),
    agent=ranking_agent,
    output_file=str(settings.outputs_dir / "ranking_output.txt"),
)