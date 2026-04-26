# feedviz/dashboard/app.py

import os
import sys
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv("feedviz/.env")

from feedviz.tools.rag_pipeline import run_rag_query
from feedviz.config.settings import settings
from feedviz.main import run_pipeline

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FeedViz — AI Teacher Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-size: 16px;
        padding: 8px;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }

    /* Main content */
    .main {
        background: #f0f4ff;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #6c63ff;
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }

    /* Teacher card */
    .teacher-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px;
        color: white;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }

    /* Top teacher card */
    .top-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 16px;
        padding: 20px;
        color: white;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(240,147,251,0.4);
    }

    /* Chat messages */
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bot {
        background: white;
        color: #333;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }

    /* Rank badge */
    .rank-badge {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 18px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
    }

    /* Progress */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    h1, h2, h3 {
        color: #302b63;
    }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
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


def data_exists():
    return (settings.outputs_dir / "overall_scores.csv").exists()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 FeedViz")
    st.markdown("*AI-Powered Teacher Analytics*")
    st.markdown("---")

    if data_exists():
        page = st.radio(
            "Navigate",
            ["🏠 Dashboard", "🏆 Rankings", "💡 Insights", "🔍 Q&A Chat"]
        )
    else:
        page = "📤 Upload"
        st.info("Upload feedback data to get started!")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "FeedViz uses AI agents, "
        "NLP, and RAG to transform "
        "student feedback into "
        "actionable insights."
    )


# ── UPLOAD PAGE ───────────────────────────────────────────────────────────────
if page == "📤 Upload" or not data_exists():
    st.markdown("# 📊 FeedViz")
    st.markdown("### AI-Powered Teacher Feedback Analytics System")
    st.markdown("---")

    st.markdown("""
    <div class='card'>
        <h3>🚀 How it works</h3>
        <p>1. Upload your student feedback CSV</p>
        <p>2. Our AI agents analyze sentiment, extract insights, and rank teachers</p>
        <p>3. View beautiful dashboards, rankings, and ask questions in natural language</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📁 Upload Feedback Data")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must have: student_id, teacher_name, department, subject, section, rating, feedback_text, date"
    )

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded: {len(df_preview)} records found!")

        with st.expander("Preview Data"):
            st.dataframe(df_preview.head(5))

        uploaded_file.seek(0)

        # Save uploaded file
        save_path = settings.raw_data_dir / "feedback_uploaded.csv"
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.markdown("### ⚙️ Required CSV Columns")
        cols = ["student_id", "teacher_name", "department",
                "subject", "section", "rating", "feedback_text", "date"]
        missing = [c for c in cols if c not in df_preview.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.success("✅ All required columns found!")

            if st.button("🚀 Analyze Feedback"):
                progress = st.progress(0)
                status   = st.empty()

                steps = [
                    "Loading and cleaning data...",
                    "Running NLP analysis...",
                    "Building vector embeddings...",
                    "Computing scores...",
                    "Ranking teachers...",
                    "Generating insights...",
                    "Pipeline complete!",
                ]

                def update_progress(step, total=len(steps)):
                    progress.progress(int((step / total) * 100))
                    status.info(f"⏳ {steps[step]}")
                    time.sleep(0.5)

                try:
                    for i in range(len(steps) - 1):
                        update_progress(i)

                    run_pipeline(str(save_path))

                    progress.progress(100)
                    status.success("✅ Analysis complete!")
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"Pipeline error: {str(e)}")


# ── DASHBOARD PAGE ────────────────────────────────────────────────────────────
elif page == "🏠 Dashboard":
    st.markdown("# 🏠 Dashboard")
    st.markdown("---")

    overall_df = load_csv(settings.outputs_dir / "overall_scores.csv")
    section_df = load_csv(settings.outputs_dir / "section_scores.csv")

    if overall_df is not None:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "👨‍🏫 Total Teachers",
            len(overall_df)
        )
        col2.metric(
            "🏆 Top Teacher",
            overall_df.loc[overall_df["overall_score"].idxmax(), "teacher_name"]
        )
        col3.metric(
            "📚 Departments",
            overall_df["department"].nunique()
        )
        col4.metric(
            "⭐ Avg Score",
            round(overall_df["overall_score"].mean(), 2)
        )

        st.markdown("---")

        # Teacher performance bar chart
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Teacher Performance")
            fig = px.bar(
                overall_df.sort_values("overall_score", ascending=True),
                x="overall_score",
                y="teacher_name",
                orientation="h",
                color="overall_score",
                color_continuous_scale="Viridis",
                title="Overall Performance Scores",
                labels={
                    "overall_score": "Score",
                    "teacher_name": "Teacher"
                }
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🏬 Department Overview")
            dept_avg = overall_df.groupby("department")["overall_score"].mean().reset_index()
            fig = px.pie(
                dept_avg,
                names="department",
                values="overall_score",
                title="Score Distribution by Department",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4,
            )
            fig.update_layout(
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Section performance
        if section_df is not None:
            st.markdown("### 📈 Section-wise Performance")
            teacher = st.selectbox(
                "Select Teacher",
                overall_df.sort_values(
                    "overall_score", ascending=False
                )["teacher_name"].tolist()
            )
            filtered = section_df[section_df["teacher_name"] == teacher]
            fig = px.bar(
                filtered,
                x="section",
                y="section_score",
                color="section",
                title=f"{teacher} — Performance by Section",
                color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#f5576c"],
                labels={"section_score": "Score", "section": "Section"}
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── RANKINGS PAGE ─────────────────────────────────────────────────────────────
elif page == "🏆 Rankings":
    st.markdown("# 🏆 Teacher Rankings")
    st.markdown("---")

    rankings = load_json(settings.rankings_output)

    if rankings:
        # Global Top 3
        st.markdown("### 🥇 Global Top 3")
        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        colors = [
            "linear-gradient(135deg, #f7971e, #ffd200)",
            "linear-gradient(135deg, #bdc3c7, #2c3e50)",
            "linear-gradient(135deg, #cd7f32, #8B4513)",
        ]
        for i, teacher in enumerate(rankings.get("global_top_3", [])):
            with cols[i]:
                st.markdown(f"""
                <div style='background:{colors[i]};border-radius:16px;
                padding:20px;text-align:center;color:white;
                box-shadow:0 4px 15px rgba(0,0,0,0.2)'>
                    <h1>{medals[i]}</h1>
                    <h3>{teacher['teacher_name']}</h3>
                    <p>{teacher['department']}</p>
                    <h2>{teacher['overall_score']}</h2>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Department leaderboards
        st.markdown("### 🏬 Department Rankings")
        dept_lb = rankings.get("department_leaderboard", {})
        for dept, teachers in dept_lb.items():
            with st.expander(f"📚 {dept}"):
                for t in teachers:
                    rank  = t.get("dept_rank", "?")
                    name  = t.get("teacher_name", "")
                    score = t.get("overall_score", 0)
                    conf  = t.get("confidence", "Normal")

                    emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                    badge = "⚠️ Low Data" if conf == "Low Confidence" else "✅"

                    st.markdown(f"""
                    <div class='teacher-card'>
                        <b>{emoji} {name}</b> &nbsp;&nbsp; {badge}
                        <br/>Score: <b>{score}</b>
                    </div>
                    """, unsafe_allow_html=True)

        # Global leaderboard
        st.markdown("### 🌍 Full Global Leaderboard")
        global_lb = rankings.get("global_leaderboard", [])
        display_df = pd.DataFrame(global_lb)[
            ["global_rank", "teacher_name", "department", "overall_score"]
        ]
        display_df.columns = ["Rank", "Teacher", "Department", "Score"]
        st.dataframe(
            display_df.style.background_gradient(
                subset=["Score"], cmap="RdYlGn"
            ),
            use_container_width=True
        )


# ── INSIGHTS PAGE ─────────────────────────────────────────────────────────────
elif page == "💡 Insights":
    st.markdown("# 💡 Teacher Insights")
    st.markdown("---")

    insights = load_json(settings.insights_output)
    overall_df = load_csv(settings.outputs_dir / "overall_scores.csv")

    if insights:
        teacher = st.selectbox(
            "Select Teacher",
            [i["teacher_name"] for i in insights]
        )
        selected = next(
            (i for i in insights if i["teacher_name"] == teacher), None
        )

        if selected and overall_df is not None:
            teacher_row = overall_df[
                overall_df["teacher_name"] == teacher
            ].iloc[0]

            # Teacher summary card
            st.markdown(f"""
            <div class='teacher-card'>
                <h2>👨‍🏫 {teacher}</h2>
                <p>Department: {teacher_row.get('department', 'N/A')} &nbsp;|&nbsp;
                   Score: {teacher_row.get('overall_score', 'N/A')} &nbsp;|&nbsp;
                   Rating: {teacher_row.get('avg_rating', 'N/A')}/5</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✅ Strengths")
                for s in selected.get("strengths", []):
                    st.success(f"• {s}")

                st.markdown("### ⚠️ Areas to Improve")
                for w in selected.get("weaknesses", []):
                    st.warning(f"• {w}")

            with col2:
                st.markdown("### 💡 Suggestions")
                for s in selected.get("suggestions", []):
                    st.info(f"• {s}")

                st.markdown("### 📝 Summary")
                st.markdown(f"""
                <div class='card'>
                    {selected.get('summary', 'No summary available')}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Run the pipeline first!")


# ── Q&A CHAT PAGE ─────────────────────────────────────────────────────────────
elif page == "🔍 Q&A Chat":
    st.markdown("# 🔍 Ask About Feedback")
    st.markdown("Ask anything about teacher performance in natural language!")
    st.markdown("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Example questions
    st.markdown("### 💬 Try asking:")
    examples = [
        "What are complaints about Dr. Rao?",
        "Who are the best teachers?",
        "What pacing issues exist?",
        "Summarize Computer Science department",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            st.session_state.messages.append({
                "role": "user",
                "content": ex
            })
            with st.spinner("Thinking..."):
                result = run_rag_query(ex, top_k=5)
                response = result.get("llm_response", "Sorry, I could not generate a response.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

    st.markdown("---")

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-user'>
                🧑 {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-bot'>
                🤖 {msg['content']}
            </div>
            """, unsafe_allow_html=True)

    # Input
    query = st.chat_input("Ask about teacher feedback...")
    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        with st.spinner("Searching and generating answer..."):
            result = run_rag_query(query, top_k=5)
            response = result.get(
                "llm_response",
                "Sorry, I could not generate a response."
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.rerun()