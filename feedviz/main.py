# feedviz/main.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv("feedviz/.env")

from feedviz.config.settings import settings
from feedviz.tools.text_cleaner import clean_feedback_dataframe
from feedviz.tools.nlp_analyzer import analyze_dataframe
from feedviz.tools.scorer import compute_teacher_scores
from feedviz.tools.ranker import (
    rank_teachers_by_department,
    rank_teachers_globally,
    get_ranking_summary,
)
from feedviz.tools.embeddings import build_faiss_index
from feedviz.tools.insight_generator import generate_all_insights


def run_pipeline(csv_path: str = None) -> dict:
    csv_path = csv_path or str(settings.feedback_csv)

    print("\n" + "="*50)
    print("  FeedViz Pipeline Starting...")
    print("="*50)

    # ── Step 1: Load & Clean ──────────────────────────
    print("\n[1/6] Loading and cleaning data...")
    df_raw = pd.read_csv(csv_path, dtype={
        "student_id":    str,
        "teacher_name":  str,
        "department":    str,
        "subject":       str,
        "section":       str,
        "feedback_text": str,
        "date":          str,
    })
    df_clean = clean_feedback_dataframe(df_raw)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(settings.cleaned_csv, index=False)
    print(f"    ✅ {len(df_clean)} records cleaned and saved")

    # ── Step 2: NLP Analysis ──────────────────────────
    print("\n[2/6] Running NLP sentiment analysis...")
    df_analyzed = analyze_dataframe(df_clean)
    analyzed_path = settings.processed_data_dir / "feedback_analyzed.csv"
    df_analyzed.to_csv(analyzed_path, index=False)
    print(f"    ✅ Sentiment, keywords, topics extracted")

    # ── Step 3: Build Embeddings ──────────────────────
    print("\n[3/6] Building FAISS vector embeddings...")
    build_faiss_index(df_analyzed)
    print(f"    ✅ FAISS index built successfully")

    # ── Step 4: Scoring ───────────────────────────────
    print("\n[4/6] Computing teacher scores...")
    section_df, overall_df = compute_teacher_scores(df_analyzed)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    section_df.to_csv(
        settings.outputs_dir / "section_scores.csv", index=False
    )
    overall_df.to_csv(
        settings.outputs_dir / "overall_scores.csv", index=False
    )
    print(f"    ✅ Section and overall scores computed")

    # ── Step 5: Ranking ───────────────────────────────
    print("\n[5/6] Ranking teachers...")
    dept_ranked   = rank_teachers_by_department(overall_df)
    global_ranked = rank_teachers_globally(dept_ranked)
    global_ranked.to_csv(
        settings.outputs_dir / "ranked_teachers.csv", index=False
    )
    ranking_summary = get_ranking_summary(global_ranked)
    with open(settings.rankings_output, "w") as f:
        json.dump(ranking_summary, f, indent=2)
    print(f"    ✅ Department and global rankings complete")

    # ── Step 6: Insights (LLM via CrewAI) ────────────
    print("\n[6/6] Generating AI insights (CrewAI + LLM)...")
    insights = generate_all_insights(global_ranked, overall_df)
    with open(settings.insights_output, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"    ✅ Insights generated for {len(insights)} teachers")

    print("\n" + "="*50)
    print("  ✅ FeedViz Pipeline Complete!")
    print("="*50 + "\n")

    return {"status": "success"}


if __name__ == "__main__":
    run_pipeline()