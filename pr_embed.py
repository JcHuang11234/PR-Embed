"""
PR-Embed: General embedding exploration utilities
Compatible with multiple conceptual pairs (e.g., CSA–CSR, AI–Human)
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine as cosine_distance
from functools import lru_cache


# =============================================================================
# A. MODEL MANAGEMENT
# =============================================================================

MODEL_DIR = "models"
META_FILE = "00_meta_w_STM.jsonl"
KEYWORD_DIR = "keywords"


def list_models():
    """List available Word2Vec models."""
    tags = []
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.startswith("word2vec_5gram_") and f.endswith(".model"):
            tags.append(f.replace("word2vec_5gram_", "").replace(".model", ""))
    return tags


@lru_cache(maxsize=5)
def load_model(tag: str):
    """Load a Word2Vec model and return KeyedVectors."""
    path = os.path.join(MODEL_DIR, f"word2vec_5gram_{tag}.model")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    #print(f"→ Loading {path}")
    return Word2Vec.load(path).wv


def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def cosine(u, v):
    if u is None or v is None:
        return np.nan
    return 1 - cosine_distance(u, v)


# =============================================================================
# B. WORD-LEVEL EXPLORATION
# =============================================================================

def word_neighbors(model, word, topn=10):
    """Return top-n most similar words."""
    if word not in model.key_to_index:
        raise ValueError(f"'{word}' not in vocabulary.")
    return model.most_similar(word, topn=topn)


def expression_neighbors(model, expression, topn=10):
    """
    Compute vector arithmetic from user expression.
    Example: "csr + authenticity - politics"
    """
    tokens = re.findall(r"[+-]?\s*\w+", expression)
    add, sub = [], []
    for t in tokens:
        t = t.strip()
        if t.startswith("-"):
            sub.append(t[1:].strip())
        else:
            add.append(t.lstrip("+").strip())

    for w in add + sub:
        if w not in model.key_to_index:
            raise ValueError(f"{w} not found in model vocab.")

    vec = np.sum([model[w] for w in add], axis=0)
    if sub:
        vec -= np.sum([model[w] for w in sub], axis=0)
    vec = normalize(vec)
    return model.similar_by_vector(vec, topn=topn)


def pair_similarity(model, w1, w2):
    """Compute cosine similarity between two words."""
    if w1 not in model.key_to_index or w2 not in model.key_to_index:
        return np.nan
    return cosine(model[w1], model[w2])


import matplotlib.pyplot as plt

def pair_similarity_over_time(w1, w2, show_plot=True):
    """
    Track cosine similarity between two words across all yearly models.
    Returns a DataFrame and optionally visualizes semantic drift.
    """
    sims = []
    for tag in list_models():
        model = load_model(tag)
        if w1 in model.key_to_index and w2 in model.key_to_index:
            sims.append({"model": tag, "cosine": pair_similarity(model, w1, w2)})
        else:
            sims.append({"model": tag, "cosine": np.nan})

    df = pd.DataFrame(sims).sort_values("model")

    if show_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(df["model"], df["cosine"], marker="o", linestyle="-", linewidth=1.8)
        plt.title(f"Semantic Similarity Over Time: {w1} ↔ {w2}", fontsize=13)
        plt.xlabel("Model (Year)", fontsize=11)
        plt.ylabel("Cosine Similarity", fontsize=11)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return df


def yearly_neighbors(word, topn=10):
    """Return neighbors for a word from all models."""
    rows = []
    for tag in list_models():
        model = load_model(tag)
        if word in model.key_to_index:
            for nb, sim in model.most_similar(word, topn=topn):
                rows.append((tag, word, nb, sim))
    return pd.DataFrame(rows, columns=["model", "word", "neighbor", "cosine"])


def yearly_drift(word, anchor):
    """Track similarity of a word to an anchor term (semantic drift)."""
    rows = []
    for tag in list_models():
        model = load_model(tag)
        if word in model.key_to_index and anchor in model.key_to_index:
            rows.append((tag, cosine(model[word], model[anchor])))
    return pd.DataFrame(rows, columns=["model", "cosine"])



# =============================================================================
# C. PAPER-LEVEL EXPLORATION
# =============================================================================

def load_metadata(path: str = META_FILE) -> pd.DataFrame:
    """
    Load metadata + STM topics (JSONL or CSV).
    Automatically detects format and handles UTF-8 encoding.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at {path}")

    if path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError("Unsupported metadata format: must be .jsonl or .csv")

    df = df.fillna("")
    df.columns = [c.lower() for c in df.columns]
    return df


def top_papers_by_topic(topic_col: str, k: int = 10, path: str = META_FILE) -> pd.DataFrame:
    """
    Return top-k papers with the highest STM proportion for a given topic column.
    Works for both JSONL and CSV metadata sources.
    """
    df = load_metadata(path)
    topic_col = topic_col.lower()

    if topic_col not in df.columns:
        available = [c for c in df.columns if c.startswith("topic_")][:5]
        raise ValueError(f"Topic column '{topic_col}' not found. Available: {available}")

    cols = [c for c in ["paper_id", "title", "year", "journal", topic_col] if c in df.columns]
    result = df.nlargest(k, topic_col)[cols].copy()
    result.rename(columns={topic_col: "topic_proportion"}, inplace=True)
    return result

def top_papers_by_word(model, word, df, k=10):
    """Find top-k papers closest to a given word in the embedding space."""
    if word not in model.key_to_index:
        raise ValueError(f"'{word}' not found in model vocabulary.")
    word_vec = model[word].reshape(1, -1)
    matrix = np.vstack(df["embedding"].values)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(word_vec, matrix)[0]
    df["similarity"] = sims
    result = df.nlargest(k, "similarity")
    keep = [c for c in ["title", "doi", "journal", "year", "similarity"] if c in result.columns]
    return result[keep].reset_index(drop=True)


def paper_topics(identifier, path: str = "00_meta_w_STM.jsonl", return_info=False):
    """
    Retrieve STM topic composition for a specific paper.
    Can match by title, DOI, URL, or alternative_id.
    If return_info=True, returns both topics and metadata.
    """
    import pandas as pd

    # Load metadata
    df = pd.read_json(path, lines=True)
    df.columns = [c.replace(".", "_") for c in df.columns]
    identifier = str(identifier).strip().lower()

    # Identify matching columns
    candidate_cols = ["paper_id", "title", "doi", "url", "alternative_id"]
    available_cols = [c for c in candidate_cols if c.lower() in [col.lower() for col in df.columns]]

    match = pd.DataFrame()
    for col in available_cols:
        col_actual = next(c for c in df.columns if c.lower() == col.lower())
        if col.lower() == "title":
            match = df[df[col_actual].str.lower().str.contains(identifier, na=False)]
        else:
            match = df[df[col_actual].astype(str).str.lower() == identifier]
        if not match.empty:
            print(f"→ Matched by '{col_actual}'")
            break

    if match.empty:
        raise ValueError(f"No paper found for identifier '{identifier}'. Checked fields: {available_cols}")

    if len(match) > 1:
        match = match.head(1)
        print("⚠️ Multiple matches found — using first result.")

    # Detect topic columns (case-insensitive)
    topic_cols = [c for c in df.columns if c.lower().startswith("topic ")]
    if not topic_cols:
        raise ValueError("No STM topic columns found. Expected columns like 'topic 1: ...'")

    # Extract topic proportions
    vals = match[topic_cols].iloc[0].to_dict()
    cleaned = {c.split(":", 1)[1].strip(): v for c, v in vals.items()}

    topics = (
        pd.DataFrame(sorted(cleaned.items(), key=lambda x: x[1], reverse=True),
                     columns=["topic", "proportion"])
        .query("proportion > 0")
        .reset_index(drop=True)
    )

    # Collect basic metadata
    paper_info = {
        "title": match["title"].iloc[0] if "title" in match.columns else None,
        "journal": match["journal"].iloc[0] if "journal" in match.columns else None,
        "year": int(match["year"].iloc[0]) if "year" in match.columns else None,
        "doi": match["doi"].iloc[0] if "doi" in match.columns else None,
        "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in match["author"].iloc[0]] if "author" in match.columns else []
    }

    return (topics, paper_info) if return_info else topics








from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def nearest_papers(identifier, df=None, path: str = META_FILE, k: int = 10):
    """
    Find the k most similar papers based on cosine similarity of embeddings.
    Works with title, DOI, URL, or alternative_id.
    Prints which field matched and displays the top results.
    """

    if df is None:
        df = load_metadata(path)

    # Normalize columns
    df.columns = [c.lower().replace(".", "_") for c in df.columns]
    if "embedding" not in df.columns:
        raise KeyError("No 'embedding' column found in metadata.")

    # Filter valid embeddings
    df = df[df["embedding"].notnull()].copy()
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=float))

    identifier = str(identifier).strip().lower()
    candidate_cols = ["paper_id", "title", "doi", "alternative_id", "url"]
    available_cols = [c for c in candidate_cols if c in df.columns]

    # Try to match input
    match = pd.DataFrame()
    for col in available_cols:
        if col == "title":
            match = df[df[col].str.lower().str.contains(identifier, na=False)]
        else:
            match = df[df[col].astype(str).str.lower() == identifier]
        if not match.empty:
            print(f"→ Matched by '{col}'")
            break

    if match.empty:
        raise ValueError(f"No paper found for '{identifier}'. Checked: {available_cols}")

    if len(match) > 1:
        match = match.head(1)
        print("⚠️ Multiple matches found — using first result.")

    # Compute cosine similarity
    target_vec = match["embedding"].iloc[0].reshape(1, -1)
    matrix = np.vstack(df["embedding"].values)
    sims = cosine_similarity(target_vec, matrix)[0]
    df["similarity"] = sims

    # Exclude the matched paper itself
    if "paper_id" in df.columns:
        target_value = match["paper_id"].iloc[0]
        mask = df["paper_id"] != target_value
    else:
        target_value = match["title"].iloc[0]
        mask = df["title"] != target_value

    # Get top-k similar papers
    result = df[mask].nlargest(k, "similarity")

    # Keep readable columns
    keep_cols = [c for c in ["title", "doi", "journal", "year", "similarity"] if c in result.columns]
    display_df = result[keep_cols].reset_index(drop=True)

    return display_df



# =============================================================================
# F. VOCABULARY & DIAGNOSTICS
# =============================================================================

def vocab_stats():
    """Vocabulary sizes and intersection across models."""
    vocabs = {tag: set(load_model(tag).key_to_index.keys()) for tag in list_models()}
    records = [(tag, len(vocabs[tag])) for tag in vocabs]
    intersect = set.intersection(*vocabs.values()) if len(vocabs) > 1 else set()
    return pd.DataFrame(records, columns=["model", "vocab_size"]), len(intersect)



# ================================================================
# WORD EQUATION PARSER + NEIGHBOR SEARCH
# ================================================================
import re
import pandas as pd

def parse_equation(eq: str):
    """
    Parse equations like 'csr + political - philanthropy'
    into positive and negative word lists.
    """
    # Normalize and remove extra spaces
    eq = eq.lower().strip()
    eq = re.sub(r"\s+", " ", eq)
    # Split by + and - with signs preserved
    tokens = re.findall(r"[+-]?\s*[\w\-]+", eq)

    positives, negatives = [], []
    for t in tokens:
        t = t.strip()
        if t.startswith("+"):
            positives.append(t[1:].strip())
        elif t.startswith("-"):
            negatives.append(t[1:].strip())
        else:
            positives.append(t.strip())
    return positives, negatives


# =============================================================================
# G. WORD EQUATION UTILITIES
# =============================================================================
import re
import pandas as pd

def parse_equation(eq: str):
    """
    Parse word equations like 'csr + political - environment'
    into positive and negative word lists.
    """
    eq = eq.lower().strip()
    eq = re.sub(r"\s+", " ", eq)
    tokens = re.findall(r"[+-]?\s*\w+", eq)

    positives, negatives = [], []
    for token in tokens:
        token = token.strip()
        if token.startswith("+"):
            word = token[1:].strip()
            if word:
                positives.append(word)
        elif token.startswith("-"):
            word = token[1:].strip()
            if word:
                negatives.append(word)
        else:
            positives.append(token)
    return positives, negatives


def safe_most_similar(model, positive, negative=None, topn=20):
    """
    Safely compute most-similar words for semantic equations.
    Handles both Word2Vec and KeyedVectors inputs.
    """
    try:
        if hasattr(model, "wv"):
            res = model.wv.most_similar(positive=positive, negative=negative or [], topn=topn)
        else:
            res = model.most_similar(positive=positive, negative=negative or [], topn=topn)
        return pd.DataFrame(res, columns=["word", "similarity"])
    except Exception as e:
        return pd.DataFrame({"word": ["N/A"], "similarity": [str(e)]})


def word_equation(model, equation: str, topn=20):
    """
    Compute vector arithmetic for equations such as:
    'csr + political - environment'
    and return nearest neighbors with cosine similarity.
    """
    pos, neg = parse_equation(equation)
    if not pos:
        raise ValueError("No valid positive terms found in equation.")

    result_df = safe_most_similar(model, pos, neg, topn)

    if neg:
        equation_str = f"{' + '.join(pos)} - {' - '.join(neg)}"
    else:
        equation_str = ' + '.join(pos)

    result_df.insert(0, "equation", equation_str)
    return result_df

