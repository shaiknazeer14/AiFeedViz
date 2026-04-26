# feedviz/dashboard/gradio_app.py

import os
import sys
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv("feedviz/.env")

from feedviz.config.settings import settings
from feedviz.tools.rag_pipeline import run_rag_query


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


# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ── */
* { box-sizing: border-box; }

body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    font-family: 'Segoe UI', sans-serif !important;
}

.gradio-container {
    background: transparent !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Glass Card ── */
.glass {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
}

/* ── Tabs ── */
.tab-nav {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    padding: 4px !important;
}

.tab-nav button {
    color: rgba(255,255,255,0.7) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.5) !important;
}

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 32px !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.5) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(102,126,234,0.7) !important;
}

/* ── Inputs ── */
input, textarea, select {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 12px 16px !important;
}

input::placeholder, textarea::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

/* ── Labels ── */
label, .label-wrap {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 600 !important;
}

/* ── Headings ── */
h1, h2, h3, h4, p, span {
    color: white !important;
}

/* ── File Upload ── */
.upload-container {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(102,126,234,0.6) !important;
    border-radius: 20px !important;
    padding: 40px !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}

.upload-container:hover {
    border-color: rgba(102,126,234,1) !important;
    background: rgba(102,126,234,0.1) !important;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 14px;
    color: rgba(255,255,255,0.6) !important;
    margin-top: 8px;
}

/* ── Teacher Cards ── */
.teacher-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
}

/* ── Chat ── */
.chatbot {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important;
}

.message.user {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border-radius: 16px 16px 4px 16px !important;
}

.message.bot {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 16px 16px 16px 4px !important;
}

/* ── Dataframe ── */
.dataframe {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ── Status ── */
.success-box {
    background: rgba(40,167,69,0.2);
    border: 1px solid rgba(40,167,69,0.5);
    border-radius: 12px;
    padding: 16px;
    color: #90EE90 !important;
}

.error-box {
    background: rgba(220,53,69,0.2);
    border: 1px solid rgba(220,53,69,0.5);
    border-radius: 12px;
    padding: 16px;
    color: #FFB6C1 !important;
}

/* ── Progress ── */
.progress-bar {
    background: linear-gradient(135deg, #667eea, #f093fb) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 3px;
}
"""


# ── PIPELINE ──────────────────────────────────────────────────────────────────
def run_analysis(file):
    if file is None:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "❌ Please upload a CSV file first!",
        )
    try:
        from feedviz.main import run_pipeline
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        save_path = settings.raw_data_dir / "feedback_uploaded.csv"
        import shutil
        shutil.copy(file.name, save_path)
        run_pipeline(str(save_path))
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "✅ Analysis complete!",
        )
    except Exception as e:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f"❌ Error: {str(e)}",
        )


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def get_dashboard():
    overall_df = load_csv(settings.outputs_dir / "overall_scores.csv")
    section_df = load_csv(settings.outputs_dir / "section_scores.csv")

    if overall_df is None:
        return None, None, "<p>No data available</p>"

    # Metrics HTML
    top    = overall_df.loc[overall_df["overall_score"].idxmax(), "teacher_name"]
    avg    = round(overall_df["overall_score"].mean(), 2)
    depts  = overall_df["department"].nunique()
    total  = len(overall_df)

    metrics_html = f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px'>
        <div class='metric-card'>
            <div class='metric-value'>{total}</div>
            <div class='metric-label'>👨‍🏫 Total Teachers</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>{avg}</div>
            <div class='metric-label'>⭐ Avg Score</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>{depts}</div>
            <div class='metric-label'>🏫 Departments</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value' style='font-size:18px'>{top}</div>
            <div class='metric-label'>🏆 Top Teacher</div>
        </div>
    </div>
    """

    # Bar chart
    fig1 = px.bar(
        overall_df.sort_values("overall_score", ascending=True),
        x="overall_score",
        y="teacher_name",
        orientation="h",
        color="overall_score",
        color_continuous_scale=[
            [0, "#f5576c"],
            [0.5, "#f093fb"],
            [1, "#667eea"]
        ],
        title="Overall Teacher Performance Scores",
        labels={"overall_score": "Score", "teacher_name": "Teacher"},
    )
    fig1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        coloraxis_showscale=False,
        title_font=dict(size=18, color="white"),
    )

    # Pie chart
    dept_avg = overall_df.groupby("department")["overall_score"].mean().reset_index()
    fig2 = px.pie(
        dept_avg,
        names="department",
        values="overall_score",
        title="Score by Department",
        color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#f5576c"],
        hole=0.5,
    )
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font=dict(size=18, color="white"),
    )

    return fig1, fig2, metrics_html


def get_section_chart(teacher_name):
    section_df = load_csv(settings.outputs_dir / "section_scores.csv")
    if section_df is None or not teacher_name:
        return None

    filtered = section_df[section_df["teacher_name"] == teacher_name]
    fig = px.bar(
        filtered,
        x="section",
        y="section_score",
        color="section",
        title=f"{teacher_name} — Section Performance",
        color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#f5576c"],
        labels={"section_score": "Score", "section": "Section"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        title_font=dict(size=16, color="white"),
    )
    return fig


# ── RANKINGS ─────────────────────────────────────────────────────────────────
def get_rankings():
    rankings = load_json(settings.rankings_output)
    if not rankings:
        return "<p>No rankings available</p>", None

    # Top 3 HTML
    medals = ["🥇", "🥈", "🥉"]
    gradients = [
        "linear-gradient(135deg,#f7971e,#ffd200)",
        "linear-gradient(135deg,#bdc3c7,#2c3e50)",
        "linear-gradient(135deg,#cd7f32,#8B4513)",
    ]
    top3_html = "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px'>"
    for i, t in enumerate(rankings.get("global_top_3", [])):
        top3_html += f"""
        <div style='background:{gradients[i]};border-radius:20px;padding:24px;
                    text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.3)'>
            <div style='font-size:48px'>{medals[i]}</div>
            <div style='font-size:20px;font-weight:800;color:white'>{t['teacher_name']}</div>
            <div style='color:rgba(255,255,255,0.8);margin:8px 0'>{t['department']}</div>
            <div style='font-size:28px;font-weight:800;color:white'>{t['overall_score']}</div>
        </div>
        """
    top3_html += "</div>"

    # Dept leaderboard HTML
    dept_lb  = rankings.get("department_leaderboard", {})
    dept_html = ""
    for dept, teachers in dept_lb.items():
        dept_html += f"""
        <div style='background:rgba(255,255,255,0.08);border-radius:16px;
                    padding:20px;margin-bottom:16px;
                    border:1px solid rgba(255,255,255,0.1)'>
            <h3 style='color:#f093fb;margin-bottom:16px'>📚 {dept}</h3>
        """
        for t in teachers:
            rank  = t.get("dept_rank", "?")
            name  = t.get("teacher_name", "")
            score = t.get("overall_score", 0)
            conf  = t.get("confidence", "Normal")
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            badge = "⚠️" if conf == "Low Confidence" else "✅"
            dept_html += f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        background:rgba(102,126,234,0.2);border-radius:12px;
                        padding:12px 16px;margin-bottom:8px'>
                <span style='font-size:18px'>{emoji} {name} {badge}</span>
                <span style='font-weight:800;color:#667eea;font-size:18px'>{score}</span>
            </div>
            """
        dept_html += "</div>"

    full_html = top3_html + dept_html

    # Global leaderboard chart
    global_lb = rankings.get("global_leaderboard", [])
    df = pd.DataFrame(global_lb)
    fig = px.bar(
        df,
        x="overall_score",
        y="teacher_name",
        orientation="h",
        color="department",
        title="Global Teacher Rankings",
        labels={"overall_score": "Score", "teacher_name": "Teacher"},
        color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#f5576c"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font=dict(size=18, color="white"),
    )

    return full_html, fig


# ── INSIGHTS ──────────────────────────────────────────────────────────────────
def get_teacher_list():
    insights = load_json(settings.insights_output)
    if not insights:
        return []
    return [i["teacher_name"] for i in insights]


def get_insight(teacher_name):
    if not teacher_name:
        return "<p>Select a teacher</p>"

    insights    = load_json(settings.insights_output)
    overall_df  = load_csv(settings.outputs_dir / "overall_scores.csv")

    if not insights:
        return "<p>No insights available. Run pipeline first.</p>"

    selected = next(
        (i for i in insights if i["teacher_name"] == teacher_name), None
    )
    if not selected:
        return "<p>Teacher not found</p>"

    teacher_row = {}
    if overall_df is not None:
        rows = overall_df[overall_df["teacher_name"] == teacher_name]
        if not rows.empty:
            teacher_row = rows.iloc[0].to_dict()

    strengths  = "".join([f"<li>✅ {s}</li>" for s in selected.get("strengths", [])])
    weaknesses = "".join([f"<li>⚠️ {w}</li>" for w in selected.get("weaknesses", [])])
    suggestions = "".join([f"<li>💡 {s}</li>" for s in selected.get("suggestions", [])])

    html = f"""
    <div style='background:linear-gradient(135deg,rgba(102,126,234,0.3),rgba(118,75,162,0.3));
                border-radius:20px;padding:24px;margin-bottom:20px;
                border:1px solid rgba(102,126,234,0.4)'>
        <h2 style='color:white;margin:0'>👨‍🏫 {teacher_name}</h2>
        <p style='color:rgba(255,255,255,0.7);margin:8px 0'>
            {teacher_row.get('department','N/A')} &nbsp;|&nbsp;
            Score: <b style='color:#f093fb'>{teacher_row.get('overall_score','N/A')}</b> &nbsp;|&nbsp;
            Rating: <b style='color:#667eea'>{teacher_row.get('avg_rating','N/A')}/5</b>
        </p>
    </div>

    <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
        <div style='background:rgba(40,167,69,0.15);border:1px solid rgba(40,167,69,0.3);
                    border-radius:16px;padding:20px'>
            <h3 style='color:#90EE90'>✅ Strengths</h3>
            <ul style='color:rgba(255,255,255,0.85);line-height:1.8'>{strengths}</ul>
        </div>
        <div style='background:rgba(255,193,7,0.15);border:1px solid rgba(255,193,7,0.3);
                    border-radius:16px;padding:20px'>
            <h3 style='color:#FFD700'>⚠️ Areas to Improve</h3>
            <ul style='color:rgba(255,255,255,0.85);line-height:1.8'>{weaknesses}</ul>
        </div>
    </div>

    <div style='background:rgba(102,126,234,0.15);border:1px solid rgba(102,126,234,0.3);
                border-radius:16px;padding:20px;margin-top:16px'>
        <h3 style='color:#667eea'>💡 Suggestions</h3>
        <ul style='color:rgba(255,255,255,0.85);line-height:1.8'>{suggestions}</ul>
    </div>

    <div style='background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);
                border-radius:16px;padding:20px;margin-top:16px'>
        <h3 style='color:white'>📝 Summary</h3>
        <p style='color:rgba(255,255,255,0.85);line-height:1.8'>
            {selected.get('summary','No summary available')}
        </p>
    </div>
    """
    return html


# ── CHAT ──────────────────────────────────────────────────────────────────────
def chat(message, history):
    if not message:
        return history, ""
    try:
        result   = run_rag_query(message, top_k=5)
        response = result.get("llm_response", "Sorry, I could not generate a response.")
    except Exception as e:
        response = f"Error: {str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""


# ── BUILD APP ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="FeedViz — AI Teacher Analytics") as app:

    # Header
    gr.HTML("""
    <div style='text-align:center;padding:40px 0 20px'>
        <h1 style='font-size:48px;font-weight:900;
                   background:linear-gradient(135deg,#667eea,#f093fb,#f5576c);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin:0'>📊 FeedViz</h1>
        <p style='color:rgba(255,255,255,0.6);font-size:18px;margin-top:8px'>
            AI-Powered Teacher Feedback Analytics
        </p>
    </div>
    """)

    # ── UPLOAD SECTION ────────────────────────────────────────────────────────
    with gr.Column(visible=True) as upload_section:
        gr.HTML("""
        <div style='max-width:600px;margin:0 auto;text-align:center'>
            <div style='background:rgba(255,255,255,0.08);backdrop-filter:blur(20px);
                        border:1px solid rgba(255,255,255,0.15);border-radius:24px;
                        padding:40px'>
                <h2 style='color:white;font-size:28px'>🚀 Get Started</h2>
                <p style='color:rgba(255,255,255,0.6);font-size:16px'>
                    Upload your student feedback CSV to begin analysis
                </p>
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="📁 Upload Feedback CSV",
                    file_types=[".csv"],
                )
                analyze_btn = gr.Button(
                    "🚀 Analyze Feedback",
                    variant="primary",
                    size="lg",
                )
                gr.HTML("""
                <div style='text-align:center;color:rgba(255,255,255,0.5);font-size:13px;margin-top:8px'>
                    ⏳ Analysis takes 2-3 minutes — please wait after clicking
                </div>
                """)
                status_box = gr.HTML("")
            with gr.Column(scale=1):
                pass

    # ── MAIN TABS ─────────────────────────────────────────────────────────────
    with gr.Column(visible=False) as main_section:

        with gr.Tabs():

            # ── DASHBOARD TAB ─────────────────────────────────────────────────
            with gr.Tab("🏠 Dashboard"):
                metrics_html  = gr.HTML()
                with gr.Row():
                    bar_chart = gr.Plot()
                    pie_chart = gr.Plot()

                gr.HTML("<h3 style='color:white'>📈 Section-wise Performance</h3>")
                overall_df = load_csv(settings.outputs_dir / "overall_scores.csv")
                teachers   = overall_df["teacher_name"].tolist() if overall_df is not None else []
                teacher_dd = gr.Dropdown(
                    choices=teachers,
                    label="Select Teacher",
                    value=teachers[0] if teachers else None
                )
                section_chart = gr.Plot()

                teacher_dd.change(
                    get_section_chart,
                    inputs=teacher_dd,
                    outputs=section_chart
                )

            # ── RANKINGS TAB ──────────────────────────────────────────────────
            with gr.Tab("🏆 Rankings"):
                rankings_html  = gr.HTML()
                rankings_chart = gr.Plot()

            # ── INSIGHTS TAB ──────────────────────────────────────────────────
            with gr.Tab("💡 Insights"):
                teacher_list = get_teacher_list()
                insight_dd   = gr.Dropdown(
                    choices=teacher_list,
                    label="Select Teacher",
                    value=teacher_list[0] if teacher_list else None
                )
                insight_html = gr.HTML()

                insight_dd.change(
                    get_insight,
                    inputs=insight_dd,
                    outputs=insight_html
                )

            # ── CHAT TAB ──────────────────────────────────────────────────────
            with gr.Tab("🔍 Q&A Chat"):
                gr.HTML("""
                <div style='margin-bottom:16px'>
                    <h3 style='color:white'>💬 Ask About Teacher Feedback</h3>
                    <p style='color:rgba(255,255,255,0.6)'>
                        Ask anything in natural language about teacher performance
                    </p>
                </div>
                """)

                chatbot = gr.Chatbot(
                    height=450,
                    show_label=False,
                    type="messages",
                    avatar_images=("👤", "🤖"),
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="e.g. What are complaints about Dr. Rao?",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

                gr.HTML("""
                <div style='margin-top:16px'>
                    <p style='color:rgba(255,255,255,0.5);font-size:13px'>
                        💡 Try: "Who are the best teachers?" |
                        "What pacing issues exist?" |
                        "Summarize Computer Science feedback"
                    </p>
                </div>
                """)

                send_btn.click(
                    chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                )
                chat_input.submit(
                    chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                )

    # ── ANALYZE BUTTON LOGIC ──────────────────────────────────────────────────

        # ── ANALYZE BUTTON LOGIC ──────────────────────────────────────────────────
        def analyze_and_load(file):
            print("[UI] Analyze button clicked!")
            print(f"[UI] File received: {file}")

            if file is None:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "<div class='error-box'>❌ Please upload a CSV file first!</div>",
                    "", None, None, None,
                    gr.Dropdown(choices=[]),
                    "", "", None,
                    gr.Dropdown(choices=[]),
                )

            try:
                import shutil
                settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
                save_path = settings.raw_data_dir / "feedback_uploaded.csv"
                shutil.copy(file.name, save_path)

                from feedviz.main import run_pipeline
                run_pipeline(str(save_path))

                f1, f2, metrics = get_dashboard()
                rhtml, rchart = get_rankings()
                teachers = get_teacher_list()
                first_insight = get_insight(teachers[0]) if teachers else ""
                sec_chart = get_section_chart(teachers[0]) if teachers else None

                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "<div class='success-box'>✅ Analysis Complete!</div>",
                    metrics or "",
                    f1, f2,
                    sec_chart,
                    gr.Dropdown(choices=teachers, value=teachers[0] if teachers else None),
                    first_insight,
                    rhtml or "",
                    rchart,
                    gr.Dropdown(choices=teachers, value=teachers[0] if teachers else None),
                )

            except Exception as e:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    f"<div class='error-box'>❌ Error: {str(e)}</div>",
                    "", None, None, None,
                    gr.Dropdown(choices=[]),
                    "", "", None,
                    gr.Dropdown(choices=[]),
                )


        analyze_btn.click(
            analyze_and_load,
            inputs=[file_input],
            outputs=[
                upload_section,
                main_section,
                status_box,
                metrics_html,
                bar_chart,
                pie_chart,
                section_chart,
                teacher_dd,
                insight_html,
                rankings_html,
                rankings_chart,
                insight_dd,
            ]
        )

    # ── LAUNCH ────────────────────────────────────────────────────────────────────
    if __name__ == "__main__":
        app.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,
            show_error=True,
            css=CUSTOM_CSS,
        )