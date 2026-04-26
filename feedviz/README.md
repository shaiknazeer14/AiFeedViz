FeedViz — AI-Powered Teacher Feedback Analytics System

Built by: Shaik Nazeer | Uttaranchal University



What is FeedViz?

FeedViz is an intelligent academic analytics system that transforms raw student feedback into actionable insights using Artificial Intelligence. Instead of manually reading thousands of feedback forms, institutions can upload a single CSV file and instantly receive teacher performance scores, department rankings, AI-generated improvement suggestions, and a natural language Q\&A interface to ask questions like "What are the common complaints about Dr. Rao?"

The system is built on a multi-agent AI architecture using CrewAI, where specialized agents handle insight generation and semantic retrieval, while optimized Python pipelines handle data processing, scoring, and ranking for maximum speed and reliability.



Problem Statement

Educational institutions collect thousands of student feedback forms every semester but lack the tools to extract meaningful insights from them. Manual review is time-consuming, inconsistent, and unscalable. There is no intelligent system that can automatically analyze feedback text, fairly rank teachers, and provide explainable improvement recommendations.



Solution

FeedViz solves this by combining three powerful AI techniques into one end-to-end system:

Natural Language Processing understands what students are saying and how they feel about each teacher. Vector Search with RAG retrieves the most relevant feedback entries when generating insights or answering questions. Multi-Agent AI uses CrewAI to reason over retrieved feedback and generate intelligent, evidence-based suggestions for each teacher.



System Architecture

The complete data flow from raw input to final output:

Student Feedback CSV

&#x20;       ↓

Data Processing    →  Clean text, remove stopwords, normalize ratings

&#x20;       ↓

NLP Analysis       →  VADER sentiment scoring, keyword extraction, topic detection

&#x20;       ↓

Embeddings         →  Convert feedback to 384-dimensional vectors

&#x20;       ↓

FAISS Vector DB    →  Store and index all embeddings for semantic search

&#x20;       ↓

Scoring            →  Weighted formula: Rating (50%) + Sentiment (30%) + Consistency (20%)

&#x20;       ↓

Ranking            →  Department-wise and global teacher leaderboards

&#x20;       ↓

Insight Agent      →  CrewAI + RAG generates strengths, weaknesses, suggestions per teacher

&#x20;       ↓

Gradio Dashboard   →  Interactive UI with charts, rankings, insights, and Q\&A chat



Key Features

Automated Feedback Analysis — Processes 15,000+ feedback records automatically after a single CSV upload. No manual configuration required.

Sentiment Analysis — Uses VADER (Valence Aware Dictionary and sEntiment Reasoner), specifically designed for short informal text like student feedback. Returns a compound score from -1 (most negative) to +1 (most positive).

Semantic Search with RAG — Converts all feedback to vector embeddings stored in FAISS. When generating insights or answering questions, the system retrieves the most semantically relevant feedback entries and passes them as context to the LLM — producing accurate, evidence-based responses instead of generic answers.

Fair Scoring System — Teacher scores are computed using a transparent weighted formula that combines student ratings, NLP sentiment, and cross-section consistency. Teachers with insufficient feedback are flagged as Low Confidence rather than excluded.

Department and Global Rankings — Teachers are ranked both within their department and globally across all departments. Gold, silver, and bronze positions are displayed for top performers.

AI-Generated Insights — For every teacher, the system generates specific strengths, areas for improvement, and actionable suggestions using the CrewAI Insight Agent powered by RAG.

Natural Language Q\&A Chat — Users can ask any question in plain English and receive intelligent answers backed by retrieved student feedback. No query language or technical knowledge required.



Tech Stack

LayerTechnologyPurposeMulti-Agent FrameworkCrewAIInsight Agent and Retrieval Agent orchestrationLLM ProviderGroq (llama-3.1-8b-instant)Fast LLM inference for insight generation and Q\&AVector DatabaseFAISSEmbedding storage and semantic similarity searchEmbeddingssentence-transformers (all-MiniLM-L6-v2)Converting feedback text to 384-dimensional vectorsSentiment AnalysisVADERNLP sentiment scoring for feedback textData ProcessingPandas, NumPyData cleaning, aggregation, and scoringDashboardGradioInteractive web UI for upload, charts, and chatBackend APIFastAPIREST endpoints for programmatic accessConfigurationpydantic-settingsType-safe settings management from .envLanguagePython 3.13Core language



Scoring Formula

Final Score = 0.5 × Average Rating (normalized)

&#x20;           + 0.3 × Sentiment Score (normalized)

&#x20;           + 0.2 × Consistency Score

Average Rating — Student-given star rating (1–5), normalized to 0–1 scale. Given highest weight as it represents the student's direct explicit judgment.

Sentiment Score — VADER compound score derived from feedback text, normalized from (-1, +1) to (0–1). Captures cases where students give polite ratings but write negative feedback.

Consistency Score — Measures how consistently a teacher performs across all sections using standard deviation. Low variation equals high consistency.



Project Structure

FeedViz/

&#x20;   feedviz/

&#x20;       agents/

&#x20;           data\_processing\_agent.py    ← Data loading and cleaning agent

&#x20;           nlp\_analysis\_agent.py       ← Sentiment and NLP agent

&#x20;           scoring\_agent.py            ← Performance scoring agent

&#x20;           ranking\_agent.py            ← Teacher ranking agent

&#x20;           insight\_agent.py            ← CrewAI insight generation agent

&#x20;           retrieval\_agent.py          ← CrewAI RAG retrieval agent

&#x20;       tools/

&#x20;           text\_cleaner.py             ← NLP preprocessing pipeline

&#x20;           nlp\_analyzer.py             ← Sentiment, keywords, topic clustering

&#x20;           embeddings.py               ← FAISS index build and search

&#x20;           rag\_pipeline.py             ← RAG query and context building

&#x20;           scorer.py                   ← Weighted scoring formula

&#x20;           ranker.py                   ← Department and global ranking

&#x20;           insight\_generator.py        ← Teacher insight generation

&#x20;       dashboard/

&#x20;           gradio\_app.py               ← Main Gradio web application

&#x20;       config/

&#x20;           settings.py                 ← Centralized configuration

&#x20;       data/

&#x20;           raw/                        ← Input CSV files

&#x20;           processed/                  ← Cleaned and analyzed CSV files

&#x20;       outputs/                        ← Scores, rankings, insights JSON

&#x20;       main.py                         ← Pipeline entry point

&#x20;       api.py                          ← FastAPI REST endpoints

&#x20;       requirements.txt                ← Python dependencies

&#x20;       .env                            ← API keys (not committed to git)



Setup Instructions

Step 1 — Clone the repository:

bashgit clone https://github.com/shaiknazeer14/AiFeedViz.git

cd AiFeedViz

Step 2 — Create and activate virtual environment:

bashpython -m venv venv

venv\\Scripts\\activate        # Windows

source venv/bin/activate     # Mac/Linux

Step 3 — Install dependencies:

bashpip install -r feedviz/requirements.txt

Step 4 — Download spaCy model:

bashpython -m spacy download en\_core\_web\_sm

Step 5 — Configure environment variables:

Create a .env file inside the feedviz/ folder:

OPENAI\_API\_KEY=your-openai-key-here

GROQ\_API\_KEY=your-groq-key-here

Get your free Groq API key at: https://console.groq.com/keys

Step 6 — Run the dashboard:

bashpython feedviz/dashboard/gradio\_app.py

Open your browser at http://localhost:7862



How to Use

Step 1 — Open the dashboard at http://localhost:7862

Step 2 — Upload your student feedback CSV file. The CSV must contain these columns:

student\_id, teacher\_name, department, subject, section, rating, feedback\_text, date

Step 3 — Click Analyze Feedback and wait 2–3 minutes for the pipeline to complete.

Step 4 — Explore your results across four tabs:



Dashboard — Overview metrics, performance bar charts, department pie chart, and section-wise analysis per teacher

Rankings — Global top 3 podium, department leaderboards, full global ranking table

Insights — Select any teacher to view AI-generated strengths, weaknesses, and improvement suggestions

Q\&A Chat — Ask natural language questions about any teacher or department





Sample Questions for Q\&A Chat

What are common complaints about Dr. Rao?

Who are the top performing teachers?

Which teachers have pacing issues?

Summarize feedback for the Computer Science department

What do students say about Dr. Sharma's teaching style?

Which sections need the most improvement?



Dataset

The system was developed and tested on a dataset of 15,000 student feedback records covering:



15 teachers across 4 departments (Computer Science, Mathematics, Electronics, Physics)

4 sections per teacher (A, B, C, D)

45 subjects

Date range: June 2023 to December 2024

Split: 10,000 training records, 5,000 test records





API Access (Optional)

FeedViz also exposes a FastAPI backend for programmatic access. After running the pipeline at least once via the dashboard:

bashuvicorn feedviz.api:app --reload --port 8000

Available endpoints:

GET  /rankings              → Returns department and global rankings

GET  /insights              → Returns AI insights for all teachers

GET  /insights/{name}       → Returns insights for a specific teacher

GET  /scores                → Returns overall teacher scores

GET  /scores/sections       → Returns section-wise scores

POST /query                 → Accepts a natural language question, returns RAG answer

Interactive API documentation available at http://localhost:8000/docs



Design Decisions

Why FAISS instead of a cloud vector database?

FAISS implements the same core concepts as cloud vector databases — embedding storage, similarity search, top-K retrieval — without requiring external infrastructure or Docker. The system architecture is designed with abstraction so the vector storage layer can be replaced with any provider when needed.

Why use CrewAI only for some agents?

Data processing, NLP analysis, scoring, and ranking are deterministic computational tasks that pure Python handles faster and more reliably than LLM reasoning. CrewAI agents are used only where AI reasoning genuinely adds value — insight generation and natural language Q\&A. This is a deliberate production engineering decision: right tool for the right job.

Why Gradio instead of Streamlit?

Gradio renders significantly faster for AI applications, provides native chat interface support, and gives better control over component visibility — essential for the upload-then-display flow in FeedViz.



Author

Shaik Nazeer

Uttaranchal University

GitHub: https://github.com/shaiknazeer14/AiFeedViz



