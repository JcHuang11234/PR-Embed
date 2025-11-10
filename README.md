# 📊 PR-Embed: Embedding Exploration Toolkit
[**🌐 Launch the App → pr-embed.streamlit.app**](https://pr-embed.streamlit.app/)

A tool for exploring conceptual and semantic structures in public relations research.  
PR-Embed integrates **semantic embeddings** and **topic modeling (STM)** to analyze how meanings, topics, and conceptual boundaries shift over time and across scholarly literature.

---

## 🧠 What Are Embeddings?

Embeddings are numerical representations of words, phrases, or documents that capture meaning from their surrounding context.  
Words that appear in similar contexts have closer vectors in a multi-dimensional semantic space.

For example, *activism*, *advocacy*, and *campaign* cluster closely together,  
while *profit* or *finance* lie farther away.

This toolkit uses **Word2Vec (Skip-gram)** embeddings trained on overlapping five-word windows (5-grams)  
to capture phrase-level semantics. The **cosine similarity** between two word vectors quantifies semantic relatedness.

> 🔗 Learn more: [A Dummy’s Guide to Word2Vec](https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673)  
> *Developed by Mikolov et al. (2013, Google Research).*

---

## 📚 Corpus and Model Training

PR-Embed includes two sets of Word2Vec models trained on full-text research articles from **six leading PR journals (2004–2024)**:

- *Public Relations Review (PRR)*  
- *Journal of Public Relations Research (JPRR)*  
- *Public Relations Inquiry (PRI)*  
- *Corporate Communications: An International Journal (CC)*  
- *Journal of Communication Management (JCM)*  
- *International Journal of Strategic Communication (IJSC)*  

**Model 1: Full-Corpus Model**  
Trained on all texts combined — captures long-term, cumulative meanings across the field.  

**Model 2: Yearly Models**  
Trained separately by publication year — ideal for tracking **semantic drift** (e.g., whether *AI* becomes more associated with *ethics* or *automation* over time).

> Each model uses identical preprocessing: lowercasing, punctuation removal, token filtering  
> (≥ 25 occurrences for the full model; ≥ 10 for yearly), and a context window size of 5.

---

## 🧩 What Is Structural Topic Modeling (STM)?

Structural Topic Modeling (STM) identifies latent topics in large text collections by grouping words that frequently co-occur.  
Each document is a mixture of topics, and each topic is a distribution of words.

| Term | Description |
|------|--------------|
| **Prob words** | Most frequent words within a topic |
| **FREX words** | Words most exclusive to that topic |
| **Topic proportion** | Share of a document devoted to a topic |

STM answers **“What are people writing about?”**  
while embeddings answer **“How are those concepts related linguistically?”**

> 🔗 Learn more: [SICSS Topic Modeling Tutorial](https://sicss.io/2020/materials/day3-text-analysis/topic-modeling/rmarkdown/Topic_Modeling.html)

---

## 🔗 How PR-Embed Combines STM and Embeddings

- **STM** provides *topic-level* insights (topics or papers as units).  
- **Embeddings** provide *word-level* insights (semantic relationships among terms).  

PR-Embed aligns STM topic vectors and Word2Vec embeddings in a shared space, enabling researchers to examine how conceptual themes (topics) relate linguistically to key ideas (words).

This integration connects **what scholars write about** (topics and papers)  
with **how those ideas relate** (semantic associations and conceptual analogies).

---

## 🧭 App Modules

### 🔤 Word-Level Exploration
Investigate how single concepts and word relationships behave.

- **Nearest Neighbors**: Finds the most similar words — and papers most aligned with a concept.  
- **Semantic Drift**: Tracks how relationships between two words (e.g., *AI* ↔ *Human*) evolve across yearly models.  
- **Word Equations**: Performs analogical reasoning (e.g., `csr + political - environment`) to derive contextual meanings.  

Use this tab to explore meaning, analogy, and conceptual evolution.

---

### 📄 Paper-Level Exploration
Search, visualize, and compare individual research papers.

- Search by **title, DOI, or URL**.  
- View a paper’s **topic composition** (as a treemap).  
- Find **semantically similar papers** in the embedding space.  

Use this tab to explore conceptual overlap among studies.

---

### 🧩 Topic-Level Exploration
Browse all **93 STM topics** derived from the PR corpus.

- Each topic lists its **Prob** and **FREX** keywords.  
- Expand a topic to see representative papers.  

Use this tab to interpret and compare thematic structures across PR research.

---

### 📈 Topic Projection
Project STM topics along any conceptual dimension (e.g., **CSA ↔ CSR**, **AI ↔ Human**).

- Select a model (e.g., *full*, *2019*, *2020*).  
- Choose two poles to define the semantic axis.  
- Compute each topic’s cosine projection.  
- Visualize results interactively with color-coded plots.  

Use this tab to explore how research themes align along theoretical continua.

---

## ⚙️ Technical Summary

| Component | Details |
|------------|----------|
| **Embedding architecture** | Skip-gram Word2Vec (Mikolov et al., 2013) |
| **Corpora** | Six leading PR journals (2004–2024) |
| **Models** | Full-corpus + yearly Word2Vec models |
| **Topic model** | Structural Topic Model (Roberts et al., 2014) with 93 topics |
| **Similarity metric** | Cosine similarity |
| **Visualization** | Streamlit + Plotly interactive interface |
| **Integration** | STM topics + embeddings for alignment |

---

## 💡 Citation

> *Anonymized for peer review.*  
> Please cite the Streamlit app:  
> **PR-Embed: Embedding Exploration Toolkit.**  
> [https://pr-embed.streamlit.app](https://pr-embed.streamlit.app)

---

## 🧩 Authors & Contributors
Anonymized for peer review

---

**⭐️ Star this repo** if you find it useful or use it for your own research!

