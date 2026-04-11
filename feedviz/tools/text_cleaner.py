# tools/text_cleaner.py
#
# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Why a separate "tool" file instead of putting this in the agent?
# ─────────────────────────────────────────────────────────────────────────────
# In CrewAI, tools are reusable units that ANY agent can use.
# By isolating text cleaning here, both the Data Processing Agent
# AND the NLP Analysis Agent can import and use the same logic.
# This follows the DRY principle (Don't Repeat Yourself).
#
# A CrewAI "tool" is just a Python function decorated with @tool.
# The decorator makes it visible to the agent's LLM so the agent
# can decide to call it during its reasoning loop.
# ─────────────────────────────────────────────────────────────────────────────

import re
import string
import pandas as pd
from typing import Optional


# ── ENGLISH STOPWORDS ────────────────────────────────────────────────────────
# WHY REMOVE STOPWORDS?
#   "The teacher is very good at explaining" →
#   "teacher good explaining"
#
#   Words like "the", "is", "at", "very" appear in EVERY sentence.
#   They carry no signal about sentiment or topic — they just add noise.
#   By removing them, we make keyword extraction and sentiment analysis
#   focus on the words that actually carry meaning.
#
# WHY A MANUAL LIST instead of nltk.corpus.stopwords?
#   Transparency + no external dependency.
#   You know exactly which words are filtered. In production,
#   you'd use NLTK's 179-word list, but this 60-word core is
#   sufficient for feedback analysis.

STOPWORDS = {
    # Articles
    "a", "an", "the",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either",
    "neither", "not", "only", "own", "same", "than", "too",
    # Pronouns
    "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their",
    # Auxiliaries
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    # Common filler
    "very", "really", "quite", "just", "also", "like",
    "this", "that", "these", "those", "which", "who",
    "what", "when", "where", "how", "all", "each",
}


# ── SIMPLE SUFFIX STEMMER ────────────────────────────────────────────────────
# WHY STEMMING/LEMMATIZATION?
#   "explains", "explained", "explaining", "explanation" all mean the same thing.
#   Without normalization, a keyword counter treats them as 4 different words.
#   With normalization → they all become "explain" → counted together.
#
# STEMMING vs LEMMATIZATION:
#   Stemming  → chops suffixes aggressively. "running" → "run", "studies" → "studi"
#               Fast, but can produce non-words.
#   Lemmatization → uses a dictionary to find the root form.
#               "running" → "run", "studies" → "study". More accurate.
#
# We implement a lightweight suffix-stripping approach here
# (NLTK's WordNetLemmatizer would be used in production).

def simple_stem(word: str) -> str:
    """
    Strip common English suffixes to get the root form.

    This is a simplified version of the Porter Stemmer algorithm.
    Each rule checks: if the word ends with suffix X and is long enough,
    strip it. Rules are ordered from most to least specific.

    Args:
        word: A single lowercase word with no punctuation.

    Returns:
        The stemmed root form.

    Examples:
        "explains"   → "explain"
        "teaching"   → "teach"
        "clearly"    → "clear"
        "structured" → "structur"  (not perfect, but consistent)
    """
    # Minimum word length after stripping — prevents "goes" → "g"
    MIN_LEN = 3

    # Each tuple: (suffix_to_remove, replacement)
    # Order matters — check longer suffixes first
    rules = [
        ("ingly", ""),
        ("ingly", ""),
        ("ation", ""),
        ("ations", ""),
        ("eness", ""),
        ("ment", ""),
        ("ments", ""),
        ("ing", ""),
        ("ings", ""),
        ("tion", ""),
        ("ions", ""),
        ("ion", ""),
        ("ies", "y"),
        ("ied", "y"),
        ("ness", ""),
        ("able", ""),
        ("ible", ""),
        ("ful", ""),
        ("less", ""),
        ("ous", ""),
        ("ive", ""),
        ("ize", ""),
        ("ise", ""),
        ("ers", ""),
        ("er", ""),
        ("est", ""),
        ("ed", ""),
        ("es", ""),
        ("ly", ""),
        ("s", ""),     # Must be last — very aggressive
    ]

    for suffix, replacement in rules:
        if word.endswith(suffix):
            root = word[: len(word) - len(suffix)] + replacement
            if len(root) >= MIN_LEN:
                return root

    return word


# ── CORE CLEANING FUNCTION ───────────────────────────────────────────────────

def clean_text(text: str, stem: bool = True) -> str:
    """
    Full NLP preprocessing pipeline for a single feedback string.

    Pipeline stages (in order):
        1. Lowercase          → "GREAT Teacher" → "great teacher"
        2. Remove URLs        → "visit https://..." → ""
        3. Remove punctuation → "great!!" → "great"
        4. Remove digits      → "section 3" → "section"
        5. Normalize spaces   → "great  teacher" → "great teacher"
        6. Tokenize           → ["great", "teacher"]
        7. Remove stopwords   → remove "the", "is", "very", etc.
        8. Stem               → "explaining" → "explain"
        9. Rejoin             → "great teacher"

    Args:
        text: Raw feedback string from the CSV.
        stem: If True, apply suffix stemming. Set False to preserve readable words.

    Returns:
        Cleaned, normalized string ready for NLP analysis.
    """

    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase
    # WHY: "Good" and "good" are the same word. Case normalization
    # ensures we don't count them separately.
    text = text.lower()

    # Step 2: Remove URLs
    # WHY: URLs like "https://..." are not meaningful feedback words.
    # \S+ matches any non-whitespace sequence starting with http/www.
    text = re.sub(r"http\S+|www\S+", "", text)

    # Step 3: Remove punctuation
    # WHY: "great!!" should become "great", not "great!!"
    # string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # We use re.sub with a character class for efficiency.
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Step 4: Remove digits
    # WHY: "section 3", "10 years" — numbers add little semantic value
    # for keyword/sentiment analysis. We keep this simple.
    text = re.sub(r"\d+", "", text)

    # Step 5: Normalize whitespace
    # WHY: After removing punctuation/digits, we get double spaces.
    # \s+ matches one or more whitespace characters (spaces, tabs, newlines).
    text = re.sub(r"\s+", " ", text).strip()

    # Step 6: Tokenize (split into words)
    # WHY: We need individual words to filter stopwords and stem.
    # Simple split() on whitespace is sufficient here.
    tokens = text.split()

    # Step 7: Remove stopwords
    # WHY: "the teacher is very good" → "teacher good"
    # We also filter out single-character tokens like "a" that
    # survived punctuation removal (e.g., from abbreviations).
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # Step 8: Stem (optional)
    # WHY: "explains", "explained", "explaining" → "explain"
    # This groups word variants together for better keyword counting.
    if stem:
        tokens = [simple_stem(t) for t in tokens]

    # Step 9: Rejoin into a single string
    # WHY: Most NLP tools (VADER, spaCy, transformers) expect a string,
    # not a list of tokens.
    return " ".join(tokens)


# ── DATAFRAME-LEVEL CLEANING ─────────────────────────────────────────────────

def clean_feedback_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full cleaning pipeline to an entire feedback DataFrame.

    DESIGN DECISION — Why keep both raw and cleaned text?
    ───────────────────────────────────────────────────────
    We NEVER delete the original data. Instead we ADD new columns:
        feedback_text         → original (for display, reports, VADER)
        feedback_clean        → stemmed (for keyword extraction, topic modeling)
        feedback_readable     → stopwords removed but NOT stemmed (for LLM insight generation)

    This way:
    - VADER sentiment runs on the ORIGINAL text (it needs punctuation + context)
    - Keyword counting uses STEMMED text (groups variants)
    - LLM insight generation uses READABLE text (not "explain fast teach")

    Args:
        df: Raw DataFrame loaded from CSV.

    Returns:
        df with new columns added, validated, and null rows handled.
    """

    df = df.copy()  # Never mutate the input DataFrame

    # ── COLUMN VALIDATION ────────────────────────────────────────────────
    required_columns = [
        "student_id", "teacher_name", "department",
        "subject", "section", "rating", "feedback_text", "date"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ── NULL HANDLING ────────────────────────────────────────────────────
    # WHY: A row with no feedback_text or no teacher_name is useless.
    # We log how many we drop rather than silently discarding them.
    original_len = len(df)
    df = df.dropna(subset=["feedback_text", "teacher_name"])
    dropped = original_len - len(df)
    if dropped > 0:
        print(f"[DataProcessingAgent] Dropped {dropped} rows with null feedback/teacher")

    # ── TYPE COERCION ────────────────────────────────────────────────────
    # WHY: CSVs store everything as strings. The scoring formula needs
    # rating as a float (1.0–5.0), and date as a datetime for trend analysis.

    # pd.to_numeric(..., errors='coerce') → invalid values become NaN
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Fill any NaN ratings with the median (not mean — median is robust to outliers)
    median_rating = df["rating"].median()
    df["rating"] = df["rating"].fillna(median_rating)

    # Clip rating to valid range [1, 5]
    # WHY: Guard against data entry errors like rating=6 or rating=0
    df["rating"] = df["rating"].clip(1, 5)

    # Parse dates — coerce invalid dates to NaT (Not a Time)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── TEXT NORMALIZATION ───────────────────────────────────────────────
    # Strip leading/trailing whitespace from string columns
    str_cols = ["teacher_name", "department", "subject", "section", "feedback_text"]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Standardize section to uppercase: "a" → "A", "b" → "B"
    # WHY: Students may enter section names in different formats (lowercase, uppercase, mixed case),
    # and some sections include alphanumeric patterns like "CSE-7", "cse-8", etc.
    # Converting to uppercase ensures consistency across all entries.
    df["section"] = df["section"].str.upper()

    # ── GENERATE CLEANED TEXT COLUMNS ───────────────────────────────────

    # feedback_clean: stemmed, stopwords removed → for keyword extraction
    df["feedback_clean"] = df["feedback_text"].apply(
        lambda x: clean_text(x, stem=True)
    )

    # feedback_readable: stopwords removed but NOT stemmed → for LLM prompts
    # WHY: An LLM reading "explain fast teach" gets confused.
    #      An LLM reading "teacher explains fast" understands perfectly.
    df["feedback_readable"] = df["feedback_text"].apply(
        lambda x: clean_text(x, stem=False)
    )

    # ── ADD HELPER COLUMNS ───────────────────────────────────────────────

    # Unique key per teacher+section combo — used heavily in aggregation
    # e.g., "Dr. Sharma_A", "Prof. Mehta_B"
    df["teacher_section_key"] = df["teacher_name"] + "_" + df["section"]

    # Normalize rating to 0–1 scale for use in scoring formula
    # WHY: The scoring formula mixes rating (1–5) with sentiment (-1 to +1).
    #      Normalizing rating to (0–1) puts all components on a comparable scale.
    df["rating_normalized"] = (df["rating"] - 1) / 4  # maps 1→0.0, 5→1.0

    return df


# ── SUMMARY STATISTICS ────────────────────────────────────────────────────────

def get_processing_summary(df: pd.DataFrame) -> dict:
    """
    Returns a summary dict after processing — useful for logging
    and for passing as context to the next agent.

    Args:
        df: Cleaned DataFrame (output of clean_feedback_dataframe).

    Returns:
        Dict with record counts, teacher list, dept list, section list.
    """
    return {
        "total_records": len(df),
        "teachers": sorted(df["teacher_name"].unique().tolist()),
        "departments": sorted(df["department"].unique().tolist()),
        "sections": sorted(df["section"].unique().tolist()),
        "subjects": sorted(df["subject"].unique().tolist()),
        "date_range": {
            "start": str(df["date"].min().date()) if not df["date"].isna().all() else "unknown",
            "end":   str(df["date"].max().date()) if not df["date"].isna().all() else "unknown",
        },
        "avg_rating_overall": round(df["rating"].mean(), 2),
        "feedback_per_teacher": df.groupby("teacher_name")["student_id"]
                                   .count()
                                   .to_dict(),
    }
