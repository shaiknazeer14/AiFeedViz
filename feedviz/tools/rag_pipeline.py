# feedviz/tools/rag_pipeline.py

import os
from dotenv import load_dotenv
load_dotenv("feedviz/.env")

from groq import Groq
from feedviz.tools.embeddings import search_similar_feedback

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_context(retrieved_feedback: list) -> str:
    """
    Converts retrieved feedback entries into a
    formatted context string for the LLM.

    Args:
        retrieved_feedback: List of dicts from search_similar_feedback()

    Returns:
        Formatted string ready to inject into LLM prompt
    """
    if not retrieved_feedback:
        return "No relevant feedback found."

    context_parts = []
    for i, entry in enumerate(retrieved_feedback, 1):
        part = (
            f"Feedback {i}:\n"
            f"  Teacher  : {entry.get('teacher_name', 'Unknown')}\n"
            f"  Subject  : {entry.get('subject', 'Unknown')}\n"
            f"  Section  : {entry.get('section', 'Unknown')}\n"
            f"  Rating   : {entry.get('rating', 'N/A')}/5\n"
            f"  Sentiment: {entry.get('sentiment_label', 'N/A')} "
            f"({entry.get('sentiment_score', 0):.2f})\n"
            f"  Topics   : {', '.join(entry.get('topics', []) or ['none'])}\n"
            f"  Feedback : {entry.get('feedback_text', '')}\n"
        )
        context_parts.append(part)

    return "\n".join(context_parts)


def run_rag_query(query: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
    1. Convert query to embedding
    2. Retrieve top-k similar feedback from FAISS
    3. Build context from retrieved feedback
    4. Pass context + query to LLM
    5. Return intelligent response

    Args:
        query: Natural language question about teacher feedback
        top_k: Number of feedback entries to retrieve

    Returns:
        Dict with retrieved feedback, context, and LLM response
    """
    # Step 1 & 2 — Retrieve relevant feedback
    print(f"[RAG] Searching for: {query}")
    retrieved = search_similar_feedback(query, top_k=top_k)

    # Step 3 — Build context
    context = build_context(retrieved)

    # Step 4 — Build prompt
    prompt = f"""You are an academic analytics assistant analyzing student feedback about teachers.

Based on the following retrieved student feedback entries, answer the query intelligently.

Retrieved Feedback:
{context}

Query: {query}

Provide a clear, structured, and actionable response. Include:
- Direct answer to the query
- Key patterns or trends observed
- Specific improvement suggestions if applicable
- Any strengths worth highlighting
"""

    # Step 5 — Call LLM
    print("[RAG] Generating response from LLM...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are an expert academic analytics assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    llm_response = response.choices[0].message.content

    return {
        "query": query,
        "retrieved_feedback": retrieved,
        "context": context,
        "llm_response": llm_response,
    }


def summarize_teacher(teacher_name: str, top_k: int = 10) -> dict:
    """
    Generates a comprehensive summary for a specific teacher
    using RAG pipeline.

    Args:
        teacher_name: Name of the teacher to summarize
        top_k: Number of feedback entries to retrieve

    Returns:
        Dict with summary and retrieved feedback
    """
    query = f"feedback performance strengths weaknesses of {teacher_name}"
    return run_rag_query(query, top_k=top_k)


def summarize_department(department: str, top_k: int = 10) -> dict:
    """
    Generates department-level feedback summary using RAG.

    Args:
        department: Department name
        top_k: Number of feedback entries to retrieve

    Returns:
        Dict with department summary
    """
    query = f"overall feedback trends issues strengths in {department} department"
    return run_rag_query(query, top_k=top_k)