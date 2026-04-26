# feedviz/tools/ranker.py

import pandas as pd


def rank_teachers_by_department(overall_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks teachers within each department based on overall_score.
    """
    overall_df = overall_df.copy()
    overall_df["dept_rank"] = overall_df.groupby("department")["overall_score"].rank(
        ascending=False, method="dense"
    ).astype(int)
    return overall_df.sort_values(["department", "dept_rank"])


def rank_teachers_globally(overall_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks all teachers across all departments globally.
    """
    overall_df = overall_df.copy()
    overall_df["global_rank"] = overall_df["overall_score"].rank(
        ascending=False, method="dense"
    ).astype(int)
    return overall_df.sort_values("global_rank")


def get_department_leaderboard(ranked_df: pd.DataFrame) -> dict:
    """
    Returns department-wise leaderboard as a dict.
    """
    leaderboard = {}
    for dept, group in ranked_df.groupby("department"):
        leaderboard[dept] = group[[
            "teacher_name", "overall_score",
            "dept_rank", "confidence"
        ]].to_dict(orient="records")
    return leaderboard


def get_global_leaderboard(ranked_df: pd.DataFrame) -> list:
    """
    Returns global leaderboard as a list sorted by global rank.
    """
    return ranked_df[[
        "global_rank", "teacher_name", "department",
        "overall_score", "confidence"
    ]].sort_values("global_rank").to_dict(orient="records")


def get_ranking_summary(ranked_df: pd.DataFrame) -> dict:
    """
    Returns complete ranking summary with both leaderboards.
    """
    dept_leaderboard   = get_department_leaderboard(ranked_df)
    global_leaderboard = get_global_leaderboard(ranked_df)

    return {
        "global_top_3": global_leaderboard[:3],
        "global_bottom_3": global_leaderboard[-3:],
        "department_leaderboard": dept_leaderboard,
        "global_leaderboard": global_leaderboard,
        "low_confidence_teachers": [
            t["teacher_name"] for t in global_leaderboard
            if t["confidence"] == "Low Confidence"
        ],
    }