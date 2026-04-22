# feedviz/tools/scorer.py

import pandas as pd
import numpy as np
from feedviz.config.settings import settings


def compute_consistency_score(ratings: list) -> float:
    """
    Consistency score based on variation across sections.
    Low variation = high consistency = good score.
    """
    if len(ratings) < 2:
        return 1.0
    std = np.std(ratings)
    max_std = 2.0
    consistency = 1.0 - min(std / max_std, 1.0)
    return round(consistency, 4)


def compute_teacher_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes section-wise and overall scores for each teacher.

    Formula:
        Final Score = 0.5 × avg_rating_normalized
                    + 0.3 × avg_sentiment_score_normalized
                    + 0.2 × consistency_score
    """
    records = []

    for (teacher, section), group in df.groupby(["teacher_name", "section"]):
        avg_rating_norm = group["rating_normalized"].mean()

        # Normalize sentiment from (-1,1) to (0,1)
        avg_sentiment = group["sentiment_score"].mean()
        avg_sentiment_norm = (avg_sentiment + 1) / 2

        final_score = (
            settings.weight_rating      * avg_rating_norm +
            settings.weight_sentiment   * avg_sentiment_norm +
            settings.weight_consistency * 1.0
        )

        records.append({
            "teacher_name":       teacher,
            "section":            section,
            "department":         group["department"].iloc[0],
            "subject":            group["subject"].iloc[0],
            "feedback_count":     len(group),
            "avg_rating":         round(group["rating"].mean(), 2),
            "avg_rating_norm":    round(avg_rating_norm, 4),
            "avg_sentiment":      round(avg_sentiment, 4),
            "avg_sentiment_norm": round(avg_sentiment_norm, 4),
            "section_score":      round(final_score, 4),
            "confidence":         "Low Confidence" if len(group) < settings.min_feedback_count else "Normal",
        })

    section_df = pd.DataFrame(records)

    # Overall teacher score across all sections
    overall_records = []
    for teacher, group in section_df.groupby("teacher_name"):
        ratings = df[df["teacher_name"] == teacher]["rating_normalized"].tolist()
        consistency = compute_consistency_score(ratings)

        avg_rating_norm   = group["avg_rating_norm"].mean()
        avg_sentiment_norm = group["avg_sentiment_norm"].mean()

        overall_score = (
            settings.weight_rating      * avg_rating_norm +
            settings.weight_sentiment   * avg_sentiment_norm +
            settings.weight_consistency * consistency
        )

        overall_records.append({
            "teacher_name":      teacher,
            "department":        group["department"].iloc[0],
            "total_feedback":    group["feedback_count"].sum(),
            "sections_taught":   list(group["section"].unique()),
            "avg_rating":        round(group["avg_rating"].mean(), 2),
            "avg_sentiment":     round(group["avg_sentiment"].mean(), 4),
            "consistency_score": round(consistency, 4),
            "overall_score":     round(overall_score, 4),
            "confidence":        "Low Confidence" if group["feedback_count"].sum() < settings.min_feedback_count else "Normal",
        })

    overall_df = pd.DataFrame(overall_records)

    return section_df, overall_df


def get_scoring_summary(section_df: pd.DataFrame, overall_df: pd.DataFrame) -> dict:
    """
    Returns scoring summary for both section and overall levels.
    """
    return {
        "total_teachers": len(overall_df),
        "top_teacher": overall_df.loc[overall_df["overall_score"].idxmax(), "teacher_name"],
        "lowest_teacher": overall_df.loc[overall_df["overall_score"].idxmin(), "teacher_name"],
        "avg_overall_score": round(overall_df["overall_score"].mean(), 4),
        "section_scores": section_df[["teacher_name", "section", "section_score"]].to_dict(orient="records"),
        "overall_scores": overall_df[["teacher_name", "overall_score", "confidence"]].to_dict(orient="records"),
    }