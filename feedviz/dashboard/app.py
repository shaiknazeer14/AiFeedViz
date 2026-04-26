# feedviz/dashboard/app.py

import os
import json
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv("feedviz/.env")

from feedviz.tools.rag_pipeline import run_rag_query
from feedviz.config.settings import settings

st.set_page_config(
    page_title="FeedViz — AI Teacher Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FeedViz — AI-Powered Teacher Feedback Analytics")
st.markdown("---")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return None


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📈 Scores", "🏆 Rankings", "💡 Insights", "🔍 Q&A Chat"]
)


# ── OVERVIEW ─────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.header("System Overview")

    overall_df = load_csv(settings.outputs_dir / "overall_scores.csv")
    rankings   = load_json(settings.rankings_output)

    if overall_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Teachers", len(overall_df))
        col2.metric("Avg Score", round(overall_df["overall_score"].mean(), 2))
        col3.metric("Top Teacher", overall_df.loc[overall_df["overall_score"].idxmax(), "teacher_name"])
        col4.metric("Departments", overall_df["department"].nunique())

        st.subheader("Score Distribution")
        fig = px.bar(
            overall_df.sort_values("overall_score", ascending=False),
            x="teacher_name", y="overall_score",
            color="department", title="Teacher Performance Scores",
            labels={"overall_score": "Score", "teacher_name": "Teacher"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run the pipeline first to generate data!")


# ── SCORES ───────────────────────────────────────────────────────────────────
elif page == "📈 Scores":
    st.header("Performance Scores")

    overall_df = load_csv(settings.outputs_dir / "overall_scores.csv")
    section_df = load_csv(settings.outputs_dir / "section_scores.csv")

    if overall_df is not None:
        st.subheader("Overall Scores")
        st.dataframe(overall_df.sort_values("overall_score", ascending=False))

        st.subheader("Score by Department")
        fig = px.box(
            overall_df, x="department", y="overall_score",
            color="department", title="Score Distribution by Department"
        )
        st.plotly_chart(fig, use_container_width=True)

    if section_df is not None:
        st.subheader("Section-wise Scores")
        teacher = st.selectbox(
            "Select Teacher",
            section_df["teacher_name"].unique()
        )
        filtered = section_df[section_df["teacher_name"] == teacher]
        fig = px.bar(
            filtered, x="section", y="section_score",
            title=f"{teacher} — Section-wise Performance",
            color="section"
        )
        st.plotly_chart(fig, use_container_width=True)


# ── RANKINGS ─────────────────────────────────────────────────────────────────
elif page == "🏆 Rankings":
    st.header("Teacher Rankings")

    rankings = load_json(settings.rankings_output)

    if rankings:
        st.subheader("🌍 Global Top 3")
        for t in rankings.get("global_top_3", []):
            st.success(
                f"#{t['global_rank']} {t['teacher_name']} "
                f"— {t['department']} — Score: {t['overall_score']}"
            )

        st.subheader("🏬 Department Leaderboards")
        dept_lb = rankings.get("department_leaderboard", {})
        for dept, teachers in dept_lb.items():
            with st.expander(f"📚 {dept}"):
                df = pd.DataFrame(teachers)
                st.dataframe(df)

        st.subheader("🌐 Global Leaderboard")
        global_lb = rankings.get("global_leaderboard", [])
        st.dataframe(pd.DataFrame(global_lb))
    else:
        st.warning("Run the pipeline first!")


# ── INSIGHTS ─────────────────────────────────────────────────────────────────
elif page == "💡 Insights":
    st.header("Teacher Insights")

    insights = load_json(settings.insights_output)

    if insights:
        teacher = st.selectbox(
            "Select Teacher",
            [i["teacher_name"] for i in insights]
        )
        selected = next(
            (i for i in insights if i["teacher_name"] == teacher), None
        )
        if selected:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ Strengths")
                for s in selected.get("strengths", []):
                    st.success(f"• {s}")

                st.subheader("⚠️ Weaknesses")
                for w in selected.get("weaknesses", []):
                    st.warning(f"• {w}")

            with col2:
                st.subheader("💡 Suggestions")
                for s in selected.get("suggestions", []):
                    st.info(f"• {s}")

                st.subheader("📝 Summary")
                st.write(selected.get("summary", ""))
    else:
        st.warning("Run the pipeline first!")


# ── Q&A CHAT ─────────────────────────────────────────────────────────────────
elif page == "🔍 Q&A Chat":
    st.header("🔍 Ask About Teacher Feedback")
    st.markdown(
        "Ask any question about teacher performance, "
        "feedback trends, or department insights."
    )

    examples = [
        "What are common complaints about Dr. Rao?",
        "Which teachers have pacing issues?",
        "Summarize feedback for Computer Science department",
        "What do students say about Dr. Sharma?",
        "Who are the best teachers overall?",
    ]

    st.subheader("Example Questions:")
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        if cols[i].button(example[:30] + "..."):
            st.session_state["query"] = example

    query = st.text_input(
        "Your Question:",
        value=st.session_state.get("query", ""),
        placeholder="e.g. What are complaints about Dr. Rao?"
    )

    if st.button("Ask", type="primary") and query:
        with st.spinner("Searching feedback and generating answer..."):
            result = run_rag_query(query, top_k=5)

        st.subheader("💬 Answer:")
        st.write(result.get("llm_response", "No response generated"))

        with st.expander("📄 Retrieved Feedback Entries"):
            for i, fb in enumerate(result.get("retrieved_feedback", []), 1):
                st.markdown(f"**{i}. {fb.get('teacher_name')}** "
                           f"({fb.get('subject')}) — "
                           f"Rating: {fb.get('rating')}/5 — "
                           f"Sentiment: {fb.get('sentiment_label')}")
                st.write(fb.get("feedback_text", ""))
                st.markdown("---")