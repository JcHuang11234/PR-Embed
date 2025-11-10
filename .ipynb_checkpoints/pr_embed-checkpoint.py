"""
PR-Embed Web App
Explore embeddings, topics, and papers interactively.
"""

import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
from pr_embed import (
    load_model, load_metadata, word_neighbors, top_papers_by_word,
    pair_similarity_over_time, top_papers_by_topic, paper_topics, nearest_papers
)

# --- Page setup ---
st.set_page_config(page_title="PR-Embed", layout="wide")

st.title("🧭 PR-Embed: Interactive Embedding Explorer")
st.markdown("""
**PR-Embed** allows you to explore conceptual relationships in public relations research  
using semantic embeddings trained on scholarly corpora.

**Core features include:**
- Word-level exploration (semantic neighbors, arithmetic, and temporal drift)
- Paper-level exploration (similarity search, topic compositions)
- Topic-level exploration (STM topics, representative papers)

Use the tabs below to start exploring.
""")

# --- Load base model & metadata once ---
@st.cache_resource
def load_all():
    model = load_model("full")
    df = load_metadata()
    return model, df

model, df = load_all()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🔤 Word-Level Exploration",
    "📄 Paper-Level Exploration",
    "🧩 Topic Exploration"
])

# --------------------------------------------------------------
# TAB 1: WORD-LEVEL
# --------------------------------------------------------------
with tab1:
    st.subheader("Explore Word Relationships")

    word = st.text_input("Enter a keyword (e.g., authenticity, csr, activism):", "authenticity")
    topn = st.slider("Number of neighbors", 5, 20, 10)
    if st.button("Show Word Neighbors"):
        try:
            neighbors = word_neighbors(model, word, topn=topn)
            st.dataframe(pd.DataFrame(neighbors, columns=["word", "similarity"]))
        except ValueError as e:
            st.warning(str(e))

    st.divider()
    st.subheader("Top Papers Related to the Word")
    if st.button("Find Top Papers"):
        try:
            papers = top_papers_by_word(model, word, df, k=10)
            st.dataframe(papers)
        except ValueError as e:
            st.warning(str(e))

    st.divider()
    st.subheader("Semantic Drift Over Time")
    w1 = st.text_input("Word 1:", "ai")
    w2 = st.text_input("Word 2:", "human")
    if st.button("Plot Semantic Drift"):
        drift = pair_similarity_over_time(w1, w2, show_plot=False)
        st.line_chart(drift.set_index("model")["cosine"])
        st.dataframe(drift)

# --------------------------------------------------------------
# TAB 2: PAPER-LEVEL
# --------------------------------------------------------------
with tab2:
    st.subheader("Find Top Papers by Topic")
    topic_name = st.text_input("Topic name or column (e.g., topic_77: stakeholder):", "topic_77: stakeholder")
    if st.button("Show Top Papers"):
        try:
            top10 = top_papers_by_topic(topic_name)
            st.dataframe(top10)
        except ValueError as e:
            st.warning(str(e))

    st.divider()
    st.subheader("Explore Paper Topics")
    identifier = st.text_input("Enter a paper title, DOI, or URL:")
    if st.button("Show Paper Topics"):
        try:
            topics = paper_topics(identifier)
            st.dataframe(topics)
            st.bar_chart(topics.set_index("topic")["proportion"])
        except ValueError as e:
            st.warning(str(e))

    st.divider()
    st.subheader("Find Similar Papers")
    if st.button("Find Similar Papers"):
        try:
            similar = nearest_papers(identifier, df=df, k=10)
            st.dataframe(similar)
        except ValueError as e:
            st.warning(str(e))

# --------------------------------------------------------------
# TAB 3: TOPIC-LEVEL
# --------------------------------------------------------------
# --------------------------------------------------------------
# TAB 3: TOPIC-LEVEL
# --------------------------------------------------------------
with tab3:
    st.subheader("Topic Exploration")
    st.markdown("""
    This tab will show topic-level relationships once topic vectors are generated.
    You can later integrate:
    - `topic_projection()` for conceptual axes
    - `affinity_table()` for group comparisons
    """)
