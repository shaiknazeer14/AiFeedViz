# feedviz/tools/rag_pipeline.py

import os
from dotenv import load_dotenv
load_dotenv("feedviz/.env")


from feedviz.tools.embeddings import search_similar_feedback

from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_context(retrieved_feedback: list) -> str:
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
            f"  Sentiment: {entry.get('sentiment_label', 'N/A')}\n"
            f"  Topics   : {', '.join(entry.get('topics', []) or ['none'])}\n"
            f"  Feedback : {entry.get('feedback_text', '')}\n"
        )
        context_parts.append(part)

    return "\n".join(context_parts)


def run_rag_query(query: str, top_k: int = 5) -> dict:
    print(f"[RAG] Searching for: {query}")
    retrieved = search_similar_feedback(query, top_k=top_k)
    context   = build_context(retrieved)

    prompt = f"""You are an academic analytics assistant analyzing student feedback about teachers.

Based on the following retrieved student feedback entries, answer the query intelligently.

Retrieved Feedback:
{context}

Query: {query}

Provide a clear, structured, and actionable response including:
- Direct answer to the query
- Key patterns or trends observed
- Specific improvement suggestions if applicable
- Any strengths worth highlighting
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert academic analytics assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    return {
        "query":              query,
        "retrieved_feedback": retrieved,
        "context":            context,
        "llm_response":       response.choices[0].message.content,
    }


def summarize_teacher(teacher_name: str, top_k: int = 10) -> dict:
    query = f"feedback performance strengths weaknesses of {teacher_name}"
    return run_rag_query(query, top_k=top_k)


def summarize_department(department: str, top_k: int = 10) -> dict:
    query = f"overall feedback trends issues strengths in {department} department"
    return run_rag_query(query, top_k=top_k)