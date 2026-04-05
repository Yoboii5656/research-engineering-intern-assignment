import streamlit as st 
import pandas as pd
import plotly.express as px
from collections import Counter
import ast
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap.umap_ as umap_reducer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load Data
df = pd.read_csv("clean.csv")
df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

# Clean Named Entities
def process_named_entities(named_entities):
    try:
        entities = ast.literal_eval(named_entities)
        return [ent for ent in entities if len(ent) > 2]  # Remove short words
    except Exception:
        return []
df["cleaned_entities"] = df["named_entities"].apply(process_named_entities)

# Semantic Search Setup
@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

encoder = load_encoder()

@st.cache_data(show_spinner="Generating Semantic Vectors (Takes ~30-60s on first load)...")
def load_embeddings(_data):
    # Combine title and selftext for richer representation, handling NaNs
    texts = (_data['title'].fillna('') + " " + _data['selftext'].fillna('')).tolist()
    return encoder.encode(texts)

embeddings = load_embeddings(df)

@st.cache_data(show_spinner="Reducing dimensions with UMAP...")
def compute_umap(_embs):
    # UMAP: 384-dim -> 2-dim, cosine metric, n_neighbors=15, min_dist=0.1
    reducer = umap_reducer.UMAP(n_components=2, metric='cosine', n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(_embs)

umap_2d = compute_umap(embeddings)

# Sentiment Analysis
df["sentiment_score"] = df["selftext"].fillna("").apply(lambda text: sia.polarity_scores(text)["compound"])
df["sentiment_label"] = df["sentiment_score"].apply(lambda score: "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral")

# Streamlit App Setup
st.set_page_config(page_title="Reddit Misinformation Dashboard", layout="wide")
st.title(" Reddit Misinformation & Trend Analysis Dashboard")

# Sidebar Filters
selected_subreddits = st.sidebar.multiselect("Select Subreddits:", options=df["subreddit"].unique(), default=["politics"])
date_range = st.sidebar.date_input("Select Date Range:", [df["created_utc"].min(), df["created_utc"].max()])
search_query = st.sidebar.text_input("Search posts:", "")

# Apply Filters
filtered_df = df[df["subreddit"].isin(selected_subreddits)]
filtered_df = filtered_df[(filtered_df["created_utc"] >= pd.to_datetime(date_range[0])) & (filtered_df["created_utc"] <= pd.to_datetime(date_range[1]))]
if search_query:
    filtered_df = filtered_df[filtered_df["selftext"].str.contains(search_query, case=False, na=False)]

# Create a tabbed layout to make the UI more interactive
tabs = st.tabs(["Misinformation Analysis", "Subreddit Engagement", "Topic Analysis", "Sentiment Analysis", "Semantic Search & Chatbot", "Topic Clustering & Embeddings"])

#  Misinformation Analysis
tabs[0].subheader("Misinformation Trends")
misinfo_counts = filtered_df.groupby(["subreddit", "misinformation_label"]).size().reset_index(name="count")
fig_misinfo = px.bar(misinfo_counts, x="subreddit", y="count", color="misinformation_label", barmode="group", title="Misinformation Trends by Subreddit")
tabs[0].plotly_chart(fig_misinfo)


# Aggregate mean engagement metrics per misinformation label
engagement_metrics = filtered_df.groupby("misinformation_label")[["num_comments", "ups"]].mean().reset_index()

# Bar chart for engagement comparison
fig_engagement = px.bar(
    engagement_metrics,
    x="misinformation_label",
    y=["num_comments", "ups"],
    barmode="group",
    title="Engagement Metrics Across Misinformation Labels",
    labels={"value": "Average Engagement", "variable": "Engagement Type"}
)

tabs[0].plotly_chart(fig_engagement)


#  Subreddit Engagement
tabs[1].subheader("Subreddit Activity & User Engagement")
engagement = filtered_df.groupby("subreddit").agg({"num_comments": "sum", "ups": "sum"}).reset_index()
fig_activity = px.bar(engagement, x="subreddit", y=["num_comments", "ups"], title="Subreddit Activity: Comments & Upvotes")
tabs[1].plotly_chart(fig_activity)

author_counts = filtered_df.groupby("subreddit")["author"].nunique().reset_index()
fig_authors = px.bar(author_counts, x="subreddit", y="author", title="Unique Authors per Subreddit")
tabs[1].plotly_chart(fig_authors)

# Adding User Activity here
tabs[1].subheader("Top Users Analysis")
top_users = filtered_df["author"].value_counts().reset_index()
top_users.columns = ["author", "post_count"]
fig_users = px.bar(top_users.head(min(10, len(top_users))), x="author", y="post_count", title="Most Active Users")
tabs[1].plotly_chart(fig_users)

#  Topic Analysis
# Count the number of posts for each topic
topic_counts = filtered_df["dominant_topic"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Count"]
# Mapping Topic Labels
topic_labels  = {
    0: "Art, Social Movements & Radical Politics",
    1: "Global Politics & Business",
    2: "US Political Discussions",
    3: "Online News & Video Content",
    4: "Anarchism, Socialism & Ideological Debates"
}
topic_counts["Topic"] = topic_counts["Topic"].map(topic_labels)

tabs[2].subheader("Topic Distribution")
# Create a pie chart
fig_topic_pie = px.pie(
    topic_counts, 
    names="Topic", 
    values="Count", 
    title="Distribution of Posts by Topic", 
    color_discrete_sequence=px.colors.qualitative.Set3
)

# Display the pie chart
tabs[2].plotly_chart(fig_topic_pie)


tabs[2].write("### Top Named Entities:")
# Flatten the list of named entities
all_entities = [ent for entities in filtered_df["cleaned_entities"] for ent in entities]

# Generate word frequency dictionary
entity_counts = Counter(all_entities)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(entity_counts)

# Display the word cloud
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")  # Hide axes

# Show the word cloud in Streamlit
tabs[2].pyplot(fig)

#  Sentiment Analysis
tabs[3].subheader("Sentiment Trends")
sentiment_counts = filtered_df.groupby(["subreddit", "sentiment_label"]).size().reset_index(name="count")
fig_sentiment = px.bar(sentiment_counts, x="subreddit", y="count", color="sentiment_label", barmode="group", title="Sentiment Distribution by Subreddit")
tabs[3].plotly_chart(fig_sentiment)

# Aggregate sentiment counts per week
filtered_df["week"] = filtered_df["created_utc"].dt.to_period("W").astype(str)
sentiment_trend = filtered_df.groupby(["week", "sentiment_label"]).size().reset_index(name="count")

# Stacked Area Chart for better visualization
fig_sentiment_trend = px.area(
    sentiment_trend, 
    x="week", 
    y="count", 
    color="sentiment_label", 
    title="Sentiment Trends Over Time (Aggregated Weekly)",
    labels={"week": "Week", "count": "Number of Posts"},
    color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"}
)

# Display updated visualization
tabs[3].plotly_chart(fig_sentiment_trend)

#  Summary & Insights
st.sidebar.write("### Summary & Insights")
st.sidebar.write("- **Misinformation trends** vary significantly across subreddits.")
st.sidebar.write("- **Engagement levels** help identify how misinformation spreads.")
st.sidebar.write("- **Topic Trends** show key discussion topics and figures.")
st.sidebar.write("- **Sentiment analysis** helps understand public opinion trends.")

#  Semantic Search & Chatbot
tabs[4].subheader("Semantic Search & Related Queries")
semantic_query = tabs[4].text_input("Ask a question or search for a topic semantically:", key="semantic_search")

if semantic_query:
    query_len = len(semantic_query.strip())
    
    # Edge case 1: Very short queries
    if query_len < 3:
        tabs[4].warning("Query too short for semantic search. Please enter at least 3 characters.")
    else:
        # Edge case 3: Non-English input gracefully handled (will yield low confidence scores if entirely nonsensical or completely unmappable, but standard multilingual inputs will map safely using vector space.)
        query_embedding = encoder.encode([semantic_query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        top_k = 5
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        # Edge case 2: Empty / low relevance results
        if top_scores[0] < 0.25: # Dynamic threshold for low relevance
            tabs[4].info("No highly relevant results found for your query. Try rephrasing!")
        else:
            tabs[4].write("### Highly Relevant Posts")
            top_texts = []
            for idx, score in zip(top_indices, top_scores):
                if score >= 0.25:
                    row = df.iloc[idx]
                    top_texts.append(str(row['title']) + " " + str(row['selftext']))
                    with tabs[4].expander(f"Score: {score:.2f} | {row['subreddit']} - {row['title']}"):
                        st.write(f"**Date:** {row['created_utc']}")
                        
                        post_text = row['selftext']
                        if pd.notna(post_text) and str(post_text).strip() != "":
                            st.write(f"**Text:** {post_text}")
                        
                        if "misinformation_label" in row:
                            st.write(f"**Misinfo Label:** {row['misinformation_label']}")
            # Chatbot: Propose related queries
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                tfidf_matrix = vectorizer.fit_transform(top_texts)
                keywords = vectorizer.get_feature_names_out()
                valid_keywords = [kw for kw in keywords if len(kw) > 3 and not kw.isnumeric()]
                
                if valid_keywords:
                    tabs[4].write("---")
                    tabs[4].write("🤖 **Chatbot suggests you might also want to explore:**")
                    cols = tabs[4].columns(min(3, len(valid_keywords)))
                    
                    def update_search(new_query):
                        st.session_state.semantic_search = new_query
                        
                    for i, kw in enumerate(valid_keywords[:3]):
                        cols[i].button(f"🔍 {kw}", key=f"suggest_{kw}", on_click=update_search, args=(kw,))
            except ValueError:
                pass # Ignore if tfidf fails (e.g. all empty strings)

# ── Topic Clustering & Embeddings ────────────────────────────────────────────
tabs[5].subheader("Topic Clustering & Embedding Visualization")
tabs[5].markdown(
    "Cluster posts by semantic similarity using **KMeans** on `all-MiniLM-L6-v2` "
    "embeddings, then visualise the 384-dim space reduced to 2-D with **UMAP**."
)

n_clusters = tabs[5].slider(
    "Number of clusters (k):", min_value=2, max_value=15, value=5, step=1,
    help="Drag to change k. Extreme values are handled gracefully."
)

@st.cache_data(show_spinner="Clustering…")
def run_kmeans(_embs, k):
    # KMeans: Euclidean distance, k-means++ init, up to 300 iterations
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    return km.fit_predict(_embs)

cluster_labels = run_kmeans(embeddings, n_clusters)
df_plot = pd.DataFrame({
    "x": umap_2d[:, 0],
    "y": umap_2d[:, 1],
    "cluster": cluster_labels.astype(str),
    "title": df["title"].fillna("(no title)"),
    "subreddit": df["subreddit"].fillna("unknown"),
})

fig_umap = px.scatter(
    df_plot, x="x", y="y", color="cluster",
    hover_data={"title": True, "subreddit": True, "x": False, "y": False},
    title=f"UMAP Embedding Space — {n_clusters} KMeans Clusters",
    labels={"cluster": "Cluster"},
    color_discrete_sequence=px.colors.qualitative.Bold,
    height=600,
)
fig_umap.update_traces(marker=dict(size=4, opacity=0.7))
fig_umap.update_layout(legend_title_text="Cluster", uirevision="umap")
tabs[5].plotly_chart(fig_umap, use_container_width=True)

# Top terms per cluster via TF-IDF
tabs[5].write("### Top Terms per Cluster")
df["_cluster"] = cluster_labels
try:
    col_pairs = tabs[5].columns(min(4, n_clusters))
    for c in range(n_clusters):
        cluster_texts = (
            df[df["_cluster"] == c]["title"].fillna("") + " " +
            df[df["_cluster"] == c]["selftext"].fillna("")
        ).tolist()
        if not cluster_texts:
            continue
        vec = TfidfVectorizer(stop_words="english", max_features=6)
        vec.fit_transform(cluster_texts)
        terms = ", ".join(vec.get_feature_names_out())
        col_pairs[c % 4].markdown(f"**Cluster {c}**\n\n{terms}")
except Exception:
    tabs[5].info("Could not extract top terms for the current cluster count.")

