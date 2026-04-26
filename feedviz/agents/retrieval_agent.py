import os
import json
from dotenv import load_dotenv
load_dotenv("feedviz/.env")

import pandas as pd
from crewai import Agent, Task, LLM
from crewai.tools import tool

from feedviz.tools.embeddings import build_faiss_index, search_similar_feedback
from feedviz.tools.rag_pipeline import run_rag_query, summarize_teacher, summarize_department
from feedviz.config.settings import settings


groq_llm = LLM(
    model="ollama/mistral",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1,
)


@tool("Build FAISS vector index from analyzed feedback")
def build_vector_index(csv_path: str) -> str:
    """
    Loads the analyzed feedback CSV, generates embeddings for all
    feedback text and stores them in a local FAISS vector index.
    Must be called before any semantic search queries.
    """
    try:
        df = pd.read_csv(csv_path)
        summary = build_faiss_index(df)
        return json.dumps(summary)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Search similar feedback using semantic query")
def semantic_search(query: str) -> str:
    """
    Converts the query into an embedding and retrieves the most
    semantically similar student feedback entries from FAISS.
    Use for questions like: What are complaints about Teacher X?
    """
    try:
        results = search_similar_feedback(query, top_k=settings.top_k_results)
        return json.dumps(results, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Run full RAG pipeline for a natural language query")
def rag_query(query: str) -> str:
    """
    Runs the complete RAG pipeline:
    1. Converts query to embedding
    2. Retrieves relevant feedback from FAISS
    3. Passes retrieved feedback as context to LLM
    4. Returns intelligent context-aware response.
    """
    try:
        result = run_rag_query(query)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Generate teacher performance summary using RAG")
def teacher_summary(teacher_name: str) -> str:
    """
    Generates a comprehensive RAG-based summary for a specific
    teacher including strengths, weaknesses, and suggestions.
    """
    try:
        result = summarize_teacher(teacher_name)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Generate department feedback summary using RAG")
def department_summary(department: str) -> str:
    """
    Generates a RAG-based summary of feedback trends
    for an entire department.
    """
    try:
        result = summarize_department(department)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


retrieval_agent = Agent(
    role="Semantic Search and RAG Specialist",
    goal=(
        "Build and manage a FAISS vector index from student feedback embeddings. "
        "Perform semantic similarity search to retrieve relevant feedback entries. "
        "Run RAG pipelines to generate intelligent context-aware insights."
    ),
    backstory=(
        "You are an expert in vector databases, semantic search, and RAG pipelines. "
        "You transform raw feedback data into a searchable vector index and use "
        "retrieval-augmented generation to produce intelligent explainable insights."
    ),
    tools=[
        build_vector_index,
        semantic_search,
        rag_query,
        teacher_summary,
        department_summary,
    ],
    llm=groq_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    use_system_prompt=True,
    respect_context_window=True,
)

retrieval_task = Task(
    description=(
        "Build a FAISS vector index from analyzed feedback at: {analyzed_csv_path}\n\n"
        "Then perform these steps:\n"
        "1. Build vector index from analyzed feedback CSV\n"
        "2. Search for: 'teachers with pacing and clarity issues'\n"
        "3. Generate summary for worst rated teacher\n"
        "4. Generate department summaries\n"
        "5. Answer: 'What are most common student complaints?'"
    ),
    expected_output=(
        "A comprehensive JSON report containing vector_index_summary, "
        "semantic_search_results, teacher_summaries, "
        "department_summaries, and common_complaints."
    ),
    agent=retrieval_agent,
    output_file=str(settings.outputs_dir / "retrieval_output.txt"),
)