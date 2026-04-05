# The Reddit Misinformation & Narrative Intelligence Dashboard

Welcome to the **Reddit Misinformation & Trend Analysis Dashboard**. The goal of this project is to go beyond simple keyword matching and surface-level analytics. We wanted to build a tool that truly understands the *context* and *semantic meaning* of discussions happening across different communities on Reddit, specifically focusing on how misinformation spreads, what topics dominate the conversation, and how users engage with these narratives.

This README will walk you through the story of our data, the analytical capabilities of the dashboard, and the advanced Machine Learning and AI components that power it.

---

## Part 1: The Core Analytics Engine

At its foundation, the dashboard ingests cleaned Reddit data (posts spanning various subreddits like `politics`, `Anarchism`, etc.) and provides interactive, tunable filtering based on date ranges and communities. 

We break the analysis down into four key pillars:

1. **Misinformation Trends:** We visualize the distribution of misinformation labels across different subreddits, alongside average engagement metrics (upvotes and comments) to see if controversial or misinformative posts garner more attention.
2. **Subreddit Engagement:** A deep dive into user activity. We track the most active authors and overall community engagement volume to identify potential "super-spreaders" or highly vocal users.
3. **Topic & Entity Analysis:** What are people actually talking about? We use topic distribution charts and dynamically generated Word Clouds based on Named Entity Recognition to highlight key figures, organizations, and themes.
4. **Sentiment Analysis:** How do people feel? By applying **VADER Sentiment Analysis**, we categorize the emotional tone of posts (Positive, Negative, Neutral) and track how these sentiments evolve over time across different communities.

---

## Part 2: The Semantic Search & Chatbot

Standard search bars use exact keyword matching (like SQL `LIKE`). We realized this wasn't enough. If a user searches for "feline companions," a strict keyword search would completely miss a post talking about "stray cats and kittens."

To solve this, we implemented a **Semantic Search Engine** backed by Sentence-Transformers. It reads the *meaning* of the user's query and finds posts with the closest conceptual alignment. 

### Zero-Keyword Overlap Examples
To prove the power of semantic search, here are 3 tested queries that share **zero lexical overlap** with the text they successfully retrieve:

* **Query:** `"feline companions"`
  * **Result Returned:** `"There's a stray cat and some kittens living near our porch."`
  * **Why it works:** The model maps "felines" to "cats" and "companions" to pets/kittens within a high-dimensional vector space.
* **Query:** `"working for zero compensation"`
  * **Result Returned:** `"My boss expects me to do unpaid volunteer labor after my shift ends."`
  * **Why it works:** "Working" conceptualizes as "labor", and "zero compensation" perfectly aligns with "unpaid volunteer."
* **Query:** `"getting physically ill"`
  * **Result Returned:** `"I am terrified of contracting a virus or catching a disease at the clinic."`
  * **Why it works:** "Ill" is semantically linked to "virus/disease", and "getting physically ill" targets "contracting/catching a virus".

### Edge Case Handling & The "Chatbot"
The semantic search is designed to be bulletproof. It handles:
* **Empty/Irrelevant Results:** If a query yields similarity scores below `0.25`, it won't return random garbage. It gracefully suggests the user rephrase.
* **Very Short Queries:** Entering $<3$ characters throws a polite warning rather than attempting a meaningless search.
* **Non-English Input:** The vectorizer processes any language. If it maps conceptually to English terms, it returns relevant results. Otherwise, it safely drops below the relevance threshold.

To make the experience more interactive, we included a **Related Query Chatbot**. Once relevant posts are found, the app uses **TF-IDF Keyword Extraction** on the top results to suggest 2-3 dynamic, related topics the user might want to click and explore next.

---

## Part 3: Topic Clustering & Embedding Visualization

Finally, we wanted a high-level topographical map of the entire conversation space. We implemented **Topic Clustering**, allowing the user to dynamically adjust the number of clusters (k) via a UI slider.

To make the high-dimensional data human-readable, we utilized **UMAP** to squish the 384-dimensional embeddings down into a 2D interactive scatter plot. You can hover over individual points to see the exact post title and subreddit, instantly revealing how specific communities cluster together around shared ideologies or topics. We further extract the "top terms" for each cluster using TF-IDF so users instantly know what that cluster is discussing.

---

## Technical Appendix: ML / AI Component Summary

To power these features, we relied on the following specific algorithms, parameters, and libraries.

| Component | Model / Algorithm | Key Parameters | Library / API |
|---|---|---|---|
| **Text Embeddings** | `all-MiniLM-L6-v2` (Sentence-BERT) | Embedding dim: 384, pooling: mean | `sentence-transformers` — `SentenceTransformer.encode()` |
| **Semantic Search** | Cosine Similarity over embeddings | Relevance threshold: 0.25, top-k: 5 | `sklearn.metrics.pairwise.cosine_similarity` |
| **Topic Clustering** | KMeans | k: tunable 2–15, init: k-means++, max_iter: 300 | `sklearn.cluster.KMeans` |
| **Embedding Visualization** | UMAP | n_components: 2, metric: cosine, n_neighbors: 15, min_dist: 0.1, random_state: 42 | `umap.umap_.UMAP.fit_transform()` |
| **Sentiment Analysis** | VADER (lexicon-based) | compound score thresholds: +0.05 / -0.05 | `nltk.sentiment.SentimentIntensityAnalyzer` |
| **Related Query Suggestions** | TF-IDF keyword extraction | max_features: 10, stop_words: english | `sklearn.feature_extraction.text.TfidfVectorizer` |
