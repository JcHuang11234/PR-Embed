from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data"
MODEL_OUT = OUT / "models"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def clean_scalar(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def export_models() -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    tags = []
    for model_path in sorted((ROOT / "models").glob("word2vec_5gram_*.model")):
        tag = model_path.stem.replace("word2vec_5gram_", "")
        print(f"Exporting model {tag}...")
        kv = Word2Vec.load(str(model_path)).wv
        words = list(kv.index_to_key)
        vectors = normalize_matrix(kv.vectors)
        vectors.astype("<f4").tofile(MODEL_OUT / f"{tag}.bin")
        write_json(
            MODEL_OUT / f"{tag}.json",
            {"tag": tag, "vectorSize": int(kv.vector_size), "words": words},
        )
        tags.append(tag)

    tags = sorted(tags, key=lambda t: (t != "full", t))
    write_json(OUT / "model_tags.json", tags)


def export_papers() -> None:
    print("Exporting paper metadata and embeddings...")
    df = pd.read_json(ROOT / "00_meta_w_STM.jsonl", lines=True)
    df.columns = [c.replace(".", "_") for c in df.columns]

    topic_cols = [c for c in df.columns if c.startswith("Topic ")]
    papers = []
    embeddings = []

    for i, row in df.reset_index(drop=True).iterrows():
        emb = np.asarray(row["embedding"], dtype="float32")
        norm = np.linalg.norm(emb)
        embeddings.append(emb / norm if norm else emb)

        authors = []
        for author in row.get("author", []) or []:
            if isinstance(author, dict):
                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                if name:
                    authors.append(name)

        topics = [
            {
                "id": int(col.split(":", 1)[0].replace("Topic", "").strip()),
                "name": col,
                "label": col.split(":", 1)[1].strip() if ":" in col else col,
                "value": float(row[col] or 0),
            }
            for col in topic_cols
        ]
        topics.sort(key=lambda x: x["value"], reverse=True)

        papers.append(
            {
                "id": int(i),
                "title": clean_scalar(row.get("title", "")),
                "journal": clean_scalar(row.get("journal", "")),
                "year": int(row.get("year", 0) or 0),
                "doi": clean_scalar(row.get("doi", "")),
                "url": clean_scalar(row.get("url", "")),
                "alternative_id": clean_scalar(row.get("alternative_id", "")),
                "authors": authors,
                "topics": topics,
            }
        )

    np.vstack(embeddings).astype("<f4").tofile(OUT / "paper_embeddings.bin")
    write_json(
        OUT / "papers.json",
        {"vectorSize": int(len(embeddings[0])), "papers": papers},
    )


def export_topics() -> None:
    print("Exporting STM topics...")
    stm = pd.read_excel(ROOT / "stm_93_done_4_plot.xlsx")
    stm["Topic"] = stm["Topic"].ffill().astype(int)
    stm["Label"] = stm["Label"].ffill().fillna("").astype(str)

    topic_rows = []
    for topic_id in sorted(stm["Topic"].unique()):
        subset = stm[stm["Topic"] == topic_id]
        label = str(subset["Label"].iloc[0])
        prob = []
        frex = []
        for _, row in subset.iterrows():
            words = [w.strip() for w in str(row["Top 20 Words"]).split(",") if w.strip()]
            word_type = str(row["Word Type"]).strip().lower()
            if word_type == "prob":
                prob = words[:20]
            elif word_type == "frex":
                frex = words[:20]
        topic_rows.append({"id": int(topic_id), "label": label, "prob": prob, "frex": frex})

    emb_df = pd.read_excel(ROOT / "stm_93_done_4_plot_with_embeddings.xlsx")
    vectors = []
    for _, row in emb_df.sort_values("Topic").iterrows():
        vec = np.fromstring(str(row["Embedding"]), sep=",", dtype="float32")
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm else vec)

    np.vstack(vectors).astype("<f4").tofile(OUT / "topic_embeddings.bin")
    write_json(
        OUT / "topics.json",
        {"vectorSize": int(len(vectors[0])), "topics": topic_rows},
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export_models()
    export_papers()
    export_topics()
    print("Done.")


if __name__ == "__main__":
    main()
