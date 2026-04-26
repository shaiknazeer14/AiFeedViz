# feedviz/tools/insight_generator.py

import os
from feedviz.tools.rag_pipeline import run_rag_query
from feedviz.config.settings import settings

from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def generate_teacher_insight(teacher_name: str, teacher_data: dict) -> dict:
    """
    Generates actionable insights for a specific teacher
    using RAG pipeline + LLM reasoning.

    Args:
        teacher_name: Name of the teacher
        teacher_data: Dict with score, sentiment, topics, rank

    Returns:
        Dict with strengths, weaknesses, and suggestions
    """
    query = f"detailed feedback complaints and strengths of {teacher_name}"
    rag_result = run_rag_query(query, top_k=8)

    prompt = f"""You are an academic performance analyst.

Teacher: {teacher_name}
Department: {teacher_data.get('department', 'N/A')}
Overall Score: {teacher_data.get('overall_score', 'N/A')}
Average Rating: {teacher_data.get('avg_rating', 'N/A')}/5
Average Sentiment: {teacher_data.get('avg_sentiment', 'N/A')}
Consistency Score: {teacher_data.get('consistency_score', 'N/A')}
Global Rank: {teacher_data.get('global_rank', 'N/A')}

Retrieved Student Feedback:
{rag_result.get('context', 'No feedback retrieved')}

Based on the above data and feedback, provide:
1. STRENGTHS: What this teacher does well (2-3 points)
2. WEAKNESSES: Areas needing improvement (2-3 points)
3. SUGGESTIONS: Specific actionable recommendations (2-3 points)
4. SUMMARY: One paragraph overall assessment

Format your response as:
STRENGTHS:
- point 1
- point 2

WEAKNESSES:
- point 1
- point 2

SUGGESTIONS:
- point 1
- point 2

SUMMARY:
paragraph here
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are an expert academic performance analyst."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    insight_text = response.choices[0].message.content

    # Parse sections
    def extract_section(text: str, section: str) -> list:
        lines = []
        in_section = False
        for line in text.split("\n"):
            if line.strip().startswith(section):
                in_section = True
                continue
            if in_section:
                if line.strip() == "" or (line.strip().endswith(":") and line.strip() != ""):
                    if lines:
                        break
                elif line.strip().startswith("-"):
                    lines.append(line.strip()[1:].strip())
        return lines

    def extract_summary(text: str) -> str:
        in_summary = False
        for line in text.split("\n"):
            if line.strip().startswith("SUMMARY:"):
                in_summary = True
                continue
            if in_summary and line.strip():
                return line.strip()
        return ""

    return {
        "teacher_name":  teacher_name,
        "strengths":     extract_section(insight_text, "STRENGTHS:"),
        "weaknesses":    extract_section(insight_text, "WEAKNESSES:"),
        "suggestions":   extract_section(insight_text, "SUGGESTIONS:"),
        "summary":       extract_summary(insight_text),
        "full_insight":  insight_text,
    }


def generate_all_insights(ranked_df, overall_df) -> list:
    """
    Generates insights for all teachers.

    Args:
        ranked_df: DataFrame with global ranks
        overall_df: DataFrame with overall scores

    Returns:
        List of insight dicts for all teachers
    """
    insights = []
    merged = ranked_df.merge(overall_df, on="teacher_name", how="left", suffixes=("", "_y"))

    for _, row in merged.iterrows():
        teacher_data = row.to_dict()
        insight = generate_teacher_insight(row["teacher_name"], teacher_data)
        insights.append(insight)
        print(f"[Insight] Generated for: {row['teacher_name']}")

    return insights