# app.py
# =============================================================================
# PR-Embed Interactive Dashboard
# =============================================================================
# Compatible with conceptual pairs such as CSA–CSR, AI–Human
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_distance
from pr_embed import (
    load_model,
    load_metadata,
    word_neighbors,
    top_papers_by_word,
    pair_similarity_over_time,
    top_papers_by_topic,
    paper_topics,
    nearest_papers,
)

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PR-Embed Explorer",
    layout="wide",
    page_icon="📊",
)

st.title("📊 PR-Embed: Embedding Exploration Toolkit")
st.markdown("""
A tool for exploring conceptual and semantic structures in Public Relations research.
Use the tabs below to navigate word-level, topic-level, and paper-level explorations.
""")

# -----------------------------------------------------------------------------
# LOAD MODELS & DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    model = load_model("full")
    
    # Load metadata with STM topic proportions
    df = pd.read_json("00_meta_w_STM.jsonl", lines=True)
    
    # Ensure embeddings are numeric arrays if present
    if "embedding" in df.columns:
        df = df[df["embedding"].notnull()].copy()
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=float))
    return model, df


model, df = load_resources()


# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "‼️ Read Me",
    "🔤 Word-Level Exploration",
    "📄 Paper-Level Exploration",
    "🧩 Topic-Level Exploration",
    "📈 Topic Projection"
])

# =============================================================================
# TAB 0: README / OVERVIEW
# =============================================================================
with tab0:
    st.markdown("""
    **PR-Embed** is an interactive system that combines **semantic embeddings** and **topic modeling (STM)**  
    to analyze conceptual relationships in public relations research.  
    It helps you explore how meanings, topics, and conceptual boundaries shift over time and across the literature.

    ---

    ### 🧠 What Are Embeddings?
    Embeddings are **numerical representations of words, phrases, or documents** that capture meaning from their surrounding context.  
    Words that frequently appear together in similar contexts have **closer vectors** in a multi-dimensional semantic space.

    For example, *activism*, *advocacy*, and *campaign* will appear close together,  
    while *profit* or *finance* will be farther away.

    This toolkit uses **Word2Vec (Skip-gram)** embeddings trained on overlapping five-word windows (5-grams)  
    to capture phrase-level semantics.  
    The **cosine similarity** between two word vectors indicates how semantically related they are  
    (high = identical meaning, low = unrelated).

    🔗 [Learn more about Word2Vec embeddings →](https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673)  
    *Developed by Mikolov et al. (2013, Google Research).*

    ---

    #### 📚 Corpus and Model Training
    PR-Embed includes **two sets of Word2Vec models** trained on peer-reviewed public relations scholarship:

    1. **Full-Corpus Model** — trained on *all full texts* from six leading PR journals between 2004 and 2024:  
       - *Public Relations Review (PRR)*  
       - *Journal of Public Relations Research (JPRR)*  
       - *Public Relations Inquiry (PRI)*  
       - *Corporate Communications: An International Journal (CC)*  
       - *Journal of Communication Management (JCM)*  
       - *International Journal of Strategic Communication (IJSC)*  
       This model captures long-term, cumulative meanings across the field.
       

    2. **Yearly Models** — trained separately for each publication year.  
       These allow you to trace **semantic drift** (how relationships between concepts evolve over time).  
       For example, you can examine whether *AI* becomes more closely associated with *ethics* or *automation* in later years.

    Each model uses identical preprocessing: lowercasing, punctuation removal, token filtering (≥ 25 occurrences for full-corpus, ≥ 10 for yearly),  
    and a context window size of 5.

    ---

    #### 🧩 What Is Structural Topic Modeling (STM)?
    STM identifies **latent topics** in large text collections by grouping words that frequently co-occur.  
    Each document is represented as a **mixture of topics**, and each topic is represented as a **distribution of words**.

    - **Prob words:** Most frequent words within a topic.  
    - **FREX words:** Words most exclusive to that topic.  
    - **Topic proportions:** How much of a document belongs to each topic.

    STM answers *“What are people writing about?”*  
    while embeddings answer *“How are those concepts related linguistically?”*

    🔗 [Learn more about STM →](https://sicss.io/2020/materials/day3-text-analysis/topic-modeling/rmarkdown/Topic_Modeling.html)  

    ---

    ### 🔗 How PR-Embed Combines STM and Embeddings
    - **STM** provides **topic-level insights**, where the unit of analysis is a *topic* or a *paper*.  
    - **Embeddings** provide **word-level insights**, focusing on how individual terms and concepts relate to one another in semantic space.  
    - **PR-Embed** embeds both **topics** (from STM) and **words/papers** (from Word2Vec) into a shared space,  
      enabling comparison of conceptual alignment and distance.  

    This integration connects what scholars **write about** (topics and documents)  
    with how those ideas **relate linguistically** (word meanings and associations).

    ---

    ### 🔤 Word-Level Exploration
    Investigate how single concepts and word relationships behave.

    - **Nearest Neighbors:** Finds the most similar words — and the papers most aligned with that concept.  
    - **Semantic Drift:** Tracks how the relationship between two words (e.g., *AI* ↔ *Human*) changes over time.  
    - **Word Equations:** Performs analogical reasoning (e.g., `csr + political − environment`)  
      to reveal derived or contextualized meanings.  

    **Use this tab to** explore meaning, analogy, and conceptual evolution.

    ---

    ### 📄 Paper-Level Exploration
    Explore how individual research papers connect to both topics and embeddings.

    - Search for a paper by **title** (or DOI/URL).  
    - View its **topic composition** (as a treemap).  
    - Identify **semantically similar papers** in the embedding space.  

    **Use this tab to** explore conceptual similarity among studies.

    ---

    ### 🧩 Topic-Level Exploration
    Browse all **93 STM topics** extracted from the PR corpus.

    - Each topic lists its **Prob** and **FREX** keywords.  
    - Expand a topic to view **representative papers** aligned with it.  

    **Use this tab to** interpret and compare thematic patterns in PR research.

    ---

    ### 📈 Topic Projection
    Project STM topics along a **semantic dimension** between two anchor words (e.g., *CSA ↔ CSR*, *AI ↔ Human*).

    - Select a model (e.g., `full`, `2019`, `2020`).  
    - Choose two poles to define the axis.  
    - Compute each topic’s **cosine projection** along the semantic continuum.  
    - Visualize the distribution interactively with color-coded bars or points.  

    **Use this tab to** see how topics align conceptually along theoretical dimensions.

    ---

    ### ⚙️ Technical Summary
    - **Embedding architecture:** Skip-gram Word2Vec (Mikolov et al., 2013).  
    - **Corpora:** Six leading PR journals (1970–2023).  
    - **Models:** One full-corpus model + yearly models for semantic drift analysis.  
    - **Topic model:** Structural Topic Model (Roberts et al., 2014) with 93 topics.  
    - **Integration:** Each topic and paper represented by an embedding vector for alignment.  
    - **Similarity metric:** Cosine similarity.  
    - **Interface:** Interactive Streamlit + Plotly visualization.  

    ---

    **Citation:**  
    Anonymized for peer review
    """)


# =============================================================================
# TAB 1: WORD-LEVEL
# =============================================================================
with tab1:
    from pr_embed import list_models, word_equation

    # --------------------------------------------------------------
    # Section 1. Single Word Exploration
    # --------------------------------------------------------------
    st.subheader("1️⃣ Single Word Exploration")
    
    try:
        vocab = sorted(list(model.key_to_index.keys()))
    except Exception:
        vocab = []
    
    word = st.selectbox(
        "Enter or select a word (type a few letters to search):",
        options=["(Type or select a word)"] + vocab,
        index=vocab.index("csa") + 1 if "csa" in vocab else 0,
        help="Start typing to find a word from the model vocabulary."
    )
    
    topn = st.slider("Number of nearest neighbors", 5, 30, 10, key="word_neighbors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Show Neighbors"):
            if word != "(Type or select a word)":
                try:
                    neighbors = pd.DataFrame(word_neighbors(model, word, topn=topn), columns=["word", "similarity"])
                    st.markdown(f"### 🔤 Nearest Neighbors of `{word}`")
    
                    import plotly.express as px
                    fig = px.bar(
                        neighbors.sort_values("similarity", ascending=False),
                        x="word",
                        y="similarity",
                        text="similarity",
                        color="similarity",
                        color_continuous_scale="Blues"
                    )
                    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    fig.update_layout(
                        xaxis_title="Word",
                        yaxis_title="Cosine Similarity",
                        xaxis_tickangle=0,
                        template="simple_white",
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(neighbors)
    
                except Exception as e:
                    st.warning(str(e))
            else:
                st.info("Please select a valid word first.")
    
    with col2:
        if st.button("📄 Show Top Papers"):
            if word != "(Type or select a word)":
                try:
                    st.markdown(f"### 📄 Top Papers Related to `{word}`")
                    papers = top_papers_by_word(model, word, df, k=10)
                    st.dataframe(papers)
                except Exception as e:
                    st.warning(str(e))
            else:
                st.info("Please select a valid word first.")

    st.divider()

    # --------------------------------------------------------------
    # Section 2. Semantic Drift (Pair of Words)
    # --------------------------------------------------------------
    st.subheader("2️⃣ Semantic Drift Between Two Words")

    w1 = st.selectbox(
        "Select Word 1:",
        options=["(Type or select a word)"] + vocab,
        index=vocab.index("csa") + 1 if "csa" in vocab else 0,
        key="w1_select",
        help="First word to compare (e.g., csa)"
    )

    w2 = st.selectbox(
        "Select Word 2:",
        options=["(Type or select a word)"] + vocab,
        index=vocab.index("csr") + 1 if "csr" in vocab else 0,
        key="w2_select",
        help="Second word to compare (e.g., csr)"
    )

    if st.button("📈 Plot Semantic Drift"):
        if "(Type or select a word)" in [w1, w2]:
            st.warning("Please select both words.")
        else:
            drift = pair_similarity_over_time(w1, w2, show_plot=False)

            st.markdown(f"### Drift of `{w1}` vs `{w2}` Across Models")
            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=drift["model"],
                y=drift["cosine"],
                mode="lines+markers",
                name="Yearly Models",
                line=dict(color="royalblue", width=3),
                marker=dict(size=8)
            ))

            try:
                full_model = load_model("full")
                baseline = full_model.similarity(w1, w2)
                fig.add_hline(
                    y=baseline,
                    line=dict(color="firebrick", width=2, dash="dash"),
                    annotation_text=f"Full model baseline = {baseline:.3f}",
                    annotation_position="top left",
                )
            except Exception:
                st.info("Full model baseline unavailable.")

            fig.update_layout(
                xaxis_title="Model (Year)",
                yaxis_title="Cosine Similarity",
                template="simple_white",
                height=400,
                margin=dict(l=40, r=30, t=50, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(drift)

    st.divider()

    # --------------------------------------------------------------
    # Section 3. Word Equation Explorer
    # --------------------------------------------------------------
    st.subheader("3️⃣ Word Equation Explorer")

    available_models = list_models()
    if "full" not in available_models:
        available_models.insert(0, "full")

    model_tag = st.selectbox(
        "Select model:",
        available_models,
        index=available_models.index("full") if "full" in available_models else 0,
        help="Select the Word2Vec model (e.g., full, 2019, 2020)"
    )

    equation_input = st.text_input(
        "Enter a word equation (e.g., csr + political - environment):",
        value="csr + political - environment",
        placeholder="csr + political - environment"
    )

    topn_eq = st.slider("Number of similar terms to display", 5, 30, 10, key="eq_topn")

    if st.button("🧮 Compute Word Equation"):
        try:
            model_selected = load_model(model_tag)
            result_df = word_equation(model_selected, equation_input, topn=topn_eq)

            st.markdown(f"**Model Used:** `{model_tag}`")
            st.dataframe(result_df)

            fig = px.bar(
                result_df.sort_values("similarity", ascending=False),
                x="word",
                y="similarity",
                text="similarity",
                color="similarity",
                color_continuous_scale="Blues"
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(
                xaxis_title="Word",
                yaxis_title="Cosine Similarity",
                xaxis_tickangle=0,
                template="simple_white",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        except FileNotFoundError:
            st.error(f"❌ Model `{model_tag}` not found in models folder.")
        except Exception as e:
            st.warning(str(e))




# =============================================================================
# TAB 2: PAPER-LEVEL (three-step exploration)
# =============================================================================
with tab2:
    import plotly.express as px

    st.subheader("Explore Paper Details")

    st.markdown("""
    Step 1: Search for a paper by title, DOI, or URL.  
    Step 2: Explore its topic composition.  
    Step 3: Find other similar papers based on semantic embeddings.
    """)

    # --- Load metadata for searching ---
    meta_path = "00_meta_w_STM.jsonl"
    df_meta = pd.read_json(meta_path, lines=True)
    df_meta.columns = [c.replace(".", "_") for c in df_meta.columns]

    # Combine fields for easier search
    df_meta["display_name"] = df_meta.apply(
        lambda x: f"{x.get('title', '')} ({x.get('year', '')}, {x.get('journal', '')})", axis=1
    )
    df_meta["identifier"] = df_meta.apply(
        lambda x: x.get("doi") or x.get("url") or x.get("title", ""), axis=1
    )

    # Step 1: Search for the paper
    options = ["🔍 Select or type to search..."] + df_meta["display_name"].tolist()
    selection = st.selectbox("Step 1: Search for a paper", options=options, index=0)

    if selection != "🔍 Select or type to search..." and selection in df_meta["display_name"].values:
        paper_row = df_meta.loc[df_meta["display_name"] == selection].iloc[0]
        paper_id = paper_row["identifier"]

        # --- Show paper metadata ---
        st.markdown("### Paper Information")
        st.markdown(f"**Title:** {paper_row.get('title', 'N/A')}")
        st.markdown(
            "**Authors:** " +
            ", ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in paper_row.get("author", [])])
        )
        st.markdown(f"**Journal:** {paper_row.get('journal', 'N/A')} ({paper_row.get('year', 'N/A')})")
        if paper_row.get("doi"):
            st.markdown(f"**DOI:** [{paper_row.get('doi')}]({paper_row.get('doi')})")
        elif paper_row.get("url"):
            st.markdown(f"**URL:** [{paper_row.get('url')}]({paper_row.get('url')})")

        # --- Step 2: Explore Topics ---
        if st.button("📊 Show Paper Topics"):
            try:
                topics, _ = paper_topics(paper_id, return_info=True)

                st.markdown("### Topic Composition")
                st.dataframe(topics)

                # Simplify labels for treemap
                topics["label"] = topics["topic"].apply(
                    lambda x: x.split(":", 1)[1].strip() if ":" in x else x
                )

                fig = px.treemap(
                    topics,
                    path=["label"],
                    values="proportion",
                    color="proportion",
                    color_continuous_scale="Blues",
                    title="Topic Composition for Selected Paper"
                )

                fig.update_traces(
                    textinfo="label+value",
                    textfont_size=14,
                    texttemplate="%{label}<br>%{value:.2%}",
                    hovertemplate="<b>%{label}</b><br>Proportion: %{value:.2%}<extra></extra>",
                )

                fig.update_layout(
                    margin=dict(t=40, l=0, r=0, b=0),
                    uniformtext=dict(minsize=12, mode="hide"),
                )

                st.plotly_chart(fig, use_container_width=True)

            except ValueError as e:
                st.warning(str(e))

        # --- Step 3: Find Similar Papers ---
        if st.button("🤝 Find Similar Papers"):
            try:
                similar = nearest_papers(paper_id, df=df_meta, k=10)
                st.markdown("### Most Similar Papers (by Embedding Similarity)")
                st.dataframe(similar)
            except ValueError as e:
                st.warning(str(e))
    else:
        st.info("Please select or search for a paper to begin.")

# =============================================================================
# TAB 3: TOPIC-LEVEL
# =============================================================================
with tab3:
    st.subheader("Topic Exploration")

    st.markdown("""
    Browse all STM topics from the PR corpus.  
    - Click a topic name to view its **representative keywords** (Prob & FREX).  
    - Inside, click again to view the **top papers** for that topic.
    """)

    stm_path = "stm_93_done_4_plot.xlsx"
    try:
        stm_df = pd.read_excel(stm_path)
    except Exception as e:
        st.error(f"Error loading {stm_path}: {e}")
        st.stop()

    stm_df["Topic"] = stm_df["Topic"].ffill().astype(int)
    stm_df["Label"] = stm_df["Label"].fillna("").astype(str)
    stm_df["topic_name"] = "Topic " + stm_df["Topic"].astype(str) + ": " + stm_df["Label"]

    topic_data = []
    for topic_id in stm_df["Topic"].unique():
        subset = stm_df[stm_df["Topic"] == topic_id]
        label = subset["Label"].iloc[0]
        prob_words, frex_words = "", ""

        for _, row in subset.iterrows():
            words = [w.strip() for w in row["Top 20 Words"].split(",")]
            if row["Word Type"].strip().lower() == "prob":
                prob_words = ", ".join(words[:20])
            elif row["Word Type"].strip().lower() == "frex":
                frex_words = ", ".join(words[:20])

        topic_data.append({
            "topic_id": topic_id,
            "label": label,
            "topic_name": f"Topic {topic_id}: {label}",
            "Prob": prob_words,
            "FREX": frex_words
        })

    topic_df = pd.DataFrame(topic_data).sort_values("topic_id").reset_index(drop=True)

    st.markdown("### All Topics")

    for _, row in topic_df.iterrows():
        with st.expander(row["topic_name"]):
            st.markdown(f"🟦 **Prob (frequent words)**: {row['Prob']}")
            st.markdown(f"🟩 **FREX (exclusive words)**: {row['FREX']}")
            st.markdown("---")

            k = st.slider(
                f"Number of top papers for {row['topic_name']}",
                5, 30, 10,
                key=f"k_{row['topic_id']}"
            )

            if st.button(f"Show Top Papers for {row['topic_name']}", key=f"btn_{row['topic_id']}"):
                try:
                    topic_prefix = f"Topic {row['topic_id']}:"
                    topic_cols = [c for c in df.columns if c.startswith(topic_prefix)]

                    if not topic_cols:
                        st.warning(f"No matching column found for {topic_prefix}.")
                    else:
                        topic_col = topic_cols[0]
                        top10 = df.nlargest(k, topic_col)[["title", "journal", "year", topic_col, "url"]]
                        st.dataframe(top10)
                except Exception as e:
                    st.warning(str(e))


# =============================================================================
# TAB 4: TOPIC PROJECTION (Semantic Dimension Explorer)
# =============================================================================
with tab4:
    import numpy as np
    import pandas as pd
    import plotly.express as px

    st.subheader("📈 Topic Projection along a Semantic Dimension")

    st.markdown("""
    Project STM topics onto a **semantic dimension** defined by any two words  
    (for example, `csa` vs `csr`, or `ai` vs `human`), using the selected Word2Vec model.
    """)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def vnorm(v):
        n = np.linalg.norm(v)
        return v / n if (n and n > 0) else v

    def phrase_vec(phrase, model):
        kv = model.wv if hasattr(model, "wv") else model
        cands = [phrase.replace(" ", "_").lower(),
                 phrase.replace(" ", "-").lower(),
                 phrase.lower()]
        for tok in cands:
            if tok in kv:
                return kv[tok]
        parts = [w for w in phrase.lower().split() if w in kv]
        return np.mean([kv[w] for w in parts], axis=0) if parts else None

    def get_vec(s, model):
        kv = model.wv if hasattr(model, "wv") else model
        s2 = s.lower()
        return kv[s2] if s2 in kv else phrase_vec(s, model)

    def cosine_projection(v, dim_unit):
        if v is None:
            return np.nan
        nv = vnorm(v)
        return float(np.dot(nv, dim_unit)) if np.linalg.norm(nv) > 0 else np.nan

    def build_dimension(pos_word, neg_word, model):
        vpos, vneg = get_vec(pos_word, model), get_vec(neg_word, model)
        if (vpos is None) or (vneg is None):
            raise ValueError(f"Missing word vectors: {pos_word}, {neg_word}")
        return vnorm(vpos - vneg)

    # ------------------------------------------------------------
    # Select model and words
    # ------------------------------------------------------------
    available_models = list_models()
    if "full" not in available_models:
        available_models.insert(0, "full")

    model_tag_dim = st.selectbox(
        "Select model:",
        available_models,
        index=available_models.index("full") if "full" in available_models else 0
    )
    model_dim = load_model(model_tag_dim)

    # Load vocab for autocompletion
    try:
        vocab = sorted(list(model_dim.key_to_index.keys()))
    except Exception:
        vocab = []

    col1, col2 = st.columns(2)
    with col1:
        pos_word = st.selectbox(
            "Positive Pole (e.g., CSA, AI):",
            options=["(Select or type a word)"] + vocab,
            index=vocab.index("csa") + 1 if "csa" in vocab else 0,
            key="pos_word_select"
        )
    with col2:
        neg_word = st.selectbox(
            "Negative Pole (e.g., CSR, Human):",
            options=["(Select or type a word)"] + vocab,
            index=vocab.index("csr") + 1 if "csr" in vocab else 0,
            key="neg_word_select"
        )

    if st.button("🚀 Project Topics"):
        if "(Select or type a word)" in [pos_word, neg_word]:
            st.warning("Please select two valid words.")
            st.stop()

        try:
            dim_vec = build_dimension(pos_word, neg_word, model_dim)
            st.success(f"Semantic dimension built: **{pos_word.upper()} → {neg_word.upper()}**")
        except Exception as e:
            st.error(str(e))
            st.stop()

        # ------------------------------------------------------------
        # Load STM topic embeddings
        # ------------------------------------------------------------
        topics_path = "stm_93_done_4_plot_with_embeddings.xlsx"
        try:
            df_topic = pd.read_excel(topics_path)
        except Exception as e:
            st.error(f"Cannot load {topics_path}: {e}")
            st.stop()

        if "Embedding" in df_topic.columns and "Topic" not in df_topic.columns:
            df_topic = df_topic.reset_index().rename(columns={"index": "Topic"})
            df_topic["Label"] = df_topic["Topic"].apply(lambda x: f"Topic {x+1}")
            df_topic["topic_vector"] = df_topic["Embedding"].apply(
                lambda x: np.array(list(map(float, str(x).split(",")))) if isinstance(x, str)
                else np.asarray(x, dtype=float)
            )
        elif "Embedding" in df_topic.columns:
            df_topic["topic_vector"] = df_topic["Embedding"].apply(
                lambda x: np.array(list(map(float, str(x).split(",")))) if isinstance(x, str)
                else np.asarray(x, dtype=float)
            )

        # ------------------------------------------------------------
        # Project topics onto the chosen dimension
        # ------------------------------------------------------------
        df_topic["projection"] = df_topic["topic_vector"].apply(lambda v: cosine_projection(v, dim_vec))
        topic_spectrum = (
            df_topic[["Label", "projection"]]
            .dropna()
            .rename(columns={"Label": "Topic"})
            .sort_values("projection", ascending=True)
            .reset_index(drop=True)
        )


        # ------------------------------------------------------------
        # Better visualization: horizontal bar chart
        # ------------------------------------------------------------
        st.markdown(f"### 🧭 Topic Spectrum Visualization: {neg_word.upper()} ↔ {pos_word.upper()}")

        fig = px.bar(
            topic_spectrum,
            y="Topic",
            x="projection",
            orientation="h",
            color="projection",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            height=1200,
            title=f"Semantic Spectrum: {neg_word.upper()} ↔ {pos_word.upper()}",
            labels={"projection": "Projection Score"},
        )

        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Projection: %{x:.3f}<extra></extra>"
        )

        fig.update_layout(
            yaxis={'categoryorder': 'array', 'categoryarray': topic_spectrum["Topic"].tolist()},
            xaxis_title=f"← {neg_word.upper()} | {pos_word.upper()} →",
            yaxis_title="",
            template="simple_white",
            coloraxis_showscale=False,
            margin=dict(l=180, r=40, t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # ------------------------------------------------------------
        # Table view
        # ------------------------------------------------------------
        st.markdown(f"### 📋 Topic Projections Table: {neg_word.upper()} ↔ {pos_word.upper()}")
        st.dataframe(
            topic_spectrum.style.format({"projection": "{:.3f}"}).background_gradient(
                cmap="RdBu_r", vmin=-1, vmax=1
            ),
            use_container_width=True,
            height=600
        )

