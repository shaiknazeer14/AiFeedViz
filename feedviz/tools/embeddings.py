# feedviz/tools/embeddings.py

import os
import json
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from feedviz.config.settings import settings

# Load embedding model once
model = SentenceTransformer(settings.embeddings_model)

# Metadata storage path (stores teacher, rating, sentiment alongside vectors)
METADATA_PATH = Path(settings.faiss_index_path).parent / "faiss_metadata.json"
INDEX_PATH = Path(settings.faiss_index_path).parent / "faiss_index.bin"


def generate_embeddings(texts: list) -> np.ndarray:
    """
    Convert list of texts into embedding vectors.
    Returns numpy array of shape (n_texts, embedding_dim)
    """
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_faiss_index(df: pd.DataFrame) -> dict:
    """
    Generates embeddings for all feedback texts and stores them in FAISS.
    Also saves metadata (teacher, subject, rating, sentiment, feedback_text)
    alongside the index for retrieval.

    Args:
        df: Analyzed DataFrame with sentiment_score, sentiment_label columns

    Returns:
        Summary dict with index size and path
    """
    # Ensure output directory exists
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[Embeddings] Generating embeddings for all feedback...")
    texts = df["feedback_text"].tolist()
    embeddings = generate_embeddings(texts)

    # FAISS requires float32
    embeddings = embeddings.astype(np.float32)

    # Get embedding dimension
    dim = embeddings.shape[1]

    # Create FAISS index
    # IndexFlatL2 = exact search using L2 (Euclidean) distance
    # For large datasets use IndexIVFFlat for approximate search
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index to disk
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[Embeddings] FAISS index saved to: {INDEX_PATH}")

    # Save metadata alongside index
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "teacher_name":    row.get("teacher_name", ""),
            "department":      row.get("department", ""),
            "subject":         row.get("subject", ""),
            "section":         row.get("section", ""),
            "rating":          float(row.get("rating", 0)),
            "sentiment_score": float(row.get("sentiment_score", 0)),
            "sentiment_label": row.get("sentiment_label", ""),
            "feedback_text":   row.get("feedback_text", ""),
            "topics":          row.get("topics", []),
        })

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    print(f"[Embeddings] Metadata saved to: {METADATA_PATH}")

    return {
        "status": "success",
        "total_vectors": index.ntotal,
        "embedding_dim": dim,
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
    }


def search_similar_feedback(query: str, top_k: int = None) -> list:
    """
    Converts a query string to embedding and retrieves
    top-k most similar feedback entries from FAISS.

    Args:
        query: Natural language query like "complaints about Dr. Rao"
        top_k: Number of results to return

    Returns:
        List of dicts with feedback text and metadata
    """
    if top_k is None:
        top_k = settings.top_k_results

    # Load FAISS index
    if not INDEX_PATH.exists():
        return [{"error": "FAISS index not found. Run build_faiss_index first."}]

    index = faiss.read_index(str(INDEX_PATH))

    # Load metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Generate query embedding
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Search FAISS
    # D = distances, I = indices of top_k results
    D, I = index.search(query_embedding, top_k)

    # Retrieve matching metadata
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(metadata):
            entry = metadata[idx].copy()
            entry["similarity_score"] = round(float(1 / (1 + dist)), 4)
            results.append(entry)

    return results