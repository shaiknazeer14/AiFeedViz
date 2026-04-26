# feedviz/agents/insight_agent.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv("feedviz/.env")

from crewai import Agent, Task, LLM
from crewai.tools import tool

from feedviz.tools.insight_generator import generate_teacher_insight, generate_all_insights
from feedviz.config.settings import settings


groq_llm = LLM(
    model="ollama/mistral",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1,
)


@tool("Generate insights for all teachers")
def generate_insights(ranked_csv_path: str) -> str:
    """
    Loads ranked teachers CSV, generates RAG-based
    insights for every teacher including strengths,
    weaknesses, and actionable suggestions.
    Saves insights to outputs directory.

    Args:
        ranked_csv_path: Path to ranked_teachers.csv

    Returns:
        JSON string with all teacher insights.
    """
    try:
        ranked_df  = pd.read_csv(ranked_csv_path)
        overall_df = pd.read_csv(settings.outputs_dir / "overall_scores.csv")

        insights = generate_all_insights(ranked_df, overall_df)

        # Save insights
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        with open(settings.insights_output, "w") as f:
            json.dump(insights, f, indent=2)

        return json.dumps({
            "status": "success",
            "total_insights": len(insights),
            "saved_to": str(settings.insights_output),
            "teachers": [i["teacher_name"] for i in insights],
        }, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Generate insight for a single teacher")
def single_teacher_insight(teacher_name: str) -> str:
    """
    Generates RAG-based insight for a single teacher.

    Args:
        teacher_name: Full name of the teacher.

    Returns:
        JSON string with teacher insight.
    """
    try:
        overall_df  = pd.read_csv(settings.outputs_dir / "overall_scores.csv")
        teacher_row = overall_df[overall_df["teacher_name"] == teacher_name]

        if teacher_row.empty:
            return json.dumps({"error": f"Teacher {teacher_name} not found"})

        teacher_data = teacher_row.iloc[0].to_dict()
        insight = generate_teacher_insight(teacher_name, teacher_data)
        return json.dumps(insight, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})


insight_agent = Agent(
    role="Academic Insight and Recommendation Specialist",
    goal=(
        "Generate intelligent, actionable insights for every teacher "
        "using RAG pipeline. Identify strengths, weaknesses, and provide "
        "specific improvement suggestions based on student feedback."
    ),
    backstory=(
        "You are an expert academic consultant who transforms raw performance "
        "data into meaningful insights. You use retrieved student feedback "
        "to generate fair, evidence-based recommendations for teachers."
    ),
    tools=[generate_insights, single_teacher_insight],
    llm=groq_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    use_system_prompt=True,
    respect_context_window=True,
)

insight_task = Task(
    description=(
        f"Generate insights for all teachers using ranked data from: "
        f"{settings.outputs_dir / 'ranked_teachers.csv'}\n\n"
        "Steps:\n"
        "1. Load ranked teachers CSV\n"
        "2. Generate RAG-based insights for each teacher\n"
        "3. Include strengths, weaknesses, suggestions\n"
        "4. Save all insights to outputs\n"
        "5. Return summary of generated insights"
    ),
    expected_output=(
        "A JSON summary containing total_insights, "
        "list of teachers processed, and saved_to path."
    ),
    agent=insight_agent,
    output_file=str(settings.outputs_dir / "insight_output.txt"),
)