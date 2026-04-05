# AI Prompts Log — Manav

I broke the project into 4 sub-parts and worked on them one by one. Used AI here and there mostly for syntax help and quick lookups — not for any design decisions.

---

## Part 1 — Data Cleaning & Preprocessing

This was the first thing i did. Raw reddit data had messy columns, stringified lists, missing values. Cleaned it in a jupyter notebook before even touching the dashboard.

---

### Prompt 1

**Component:** Named entity parsing (`cleaning.ipynb`)

**Prompt:**
> "pandas parse stringified list column and extract just the text values"

**What was wrong / how I fixed it:**
Output used `json.loads()` which crashed because reddit data uses single quotes not double quotes. Switched to `ast.literal_eval()` inside a try-except manually.

---

### Prompt 2

**Component:** Filtering short/noisy entities (`cleaning.ipynb`)

**Prompt:**
> "how to filter items in a list inside a pandas column based on string length greater than 2"

**What was wrong / how I fixed it:**
The AI gave a correct snippet but applied it on the original column instead of the already-parsed one. I had to reorganize the order of operations myself — parse first, then filter.

---

## Part 2 — Core Dashboard (Streamlit Structure + Charts)

After data was clean I started building the actual dashboard. Set up the tab layout, sidebar filters, and the main charts first before adding any ML stuff.

---

### Prompt 3

**Component:** Tab layout and sidebar (`app.py`)

**Prompt:**
> "streamlit multi tab layout with sidebar filters subreddit and date range"

**What was wrong / how I fixed it:**
The example used `with st.sidebar:` block for everything which is fine for simple apps but when nesting complex widgets it got messy. I moved to calling `st.sidebar.header()` and `st.sidebar.multiselect()` directly which gave me more control.

---

### Prompt 4

**Component:** Grouped bar chart for misinformation trends (`app.py`)

**Prompt:**
> "plotly grouped bar chart from a pandas groupby result with two categories on color"

**What was wrong / how I fixed it:**
AI's groupby used `.pivot()` before passing to plotly which added an extra step I didn't need. Plotly express handles the `color=` parameter directly from a long-format df — I just removed the pivot and passed the groupby reset_index output directly.

---

### Prompt 5

**Component:** Sentiment labeling (`app.py`)

**Prompt:**
> "nltk vader sentiment on dataframe column, add label column positive negative neutral"

**What was wrong / how I fixed it:**
Thresholds were set at `> 0` and `< 0` which is wrong — VADER docs say to use 0.05 and -0.05 to avoid labeling weakly scored text. Fixed the thresholds myself after reading the VADER paper.

---

## Part 3 — ML Features (Semantic Search, Clustering, UMAP)

This was the hardest part. I had a general idea of how semantic search and clustering would work but hadn't used sentence-transformers with streamlit before so needed some help with the caching patterns.

---

### Prompt 6

**Component:** Sentence transformer caching in streamlit (`app.py`)

**Prompt:**
> "how to cache a sentence transformer model in streamlit so it doesn't reload every time"

**What was wrong / how I fixed it:**
AI said to use `@st.cache_data` on the model loader but that failed with a hash error because the SentenceTransformer object cant be serialized. I had to look up Streamlit docs myself and found `@st.cache_resource` is the right decorator for stateful objects like ML models. Used `@st.cache_data` only for the embeddings array, and prefixed the arg with `_` to skip hashing it.

---

### Prompt 7

**Component:** Cosine similarity top-k retrieval (`app.py`)

**Prompt:**
> "cosine similarity between one query vector and a matrix of vectors get top 5 indexes python"

**What was wrong / how I fixed it:**
Output was mostly fine but sorted ascending instead of descending so it returned the worst matches. Added `[::-1]` to the argsort myself.

---

### Prompt 8

**Component:** UMAP configuration (`app.py`)

**Prompt:**
> "umap settings for sentence embeddings whats good n_neighbors value"

**What was wrong / how I fixed it:**
Suggested `n_neighbors=5` which made clusters look scattered and noisy when i tested it visually. Bumped it to 15 after reading the UMAP docs. Also AI forgot to set `metric='cosine'` which is important because embeddings arent euclidean — added that myself.

---

### Prompt 9

**Component:** KMeans on embeddings with streamlit slider (`app.py`)

**Prompt:**
> "kmeans clustering on numpy array with variable k, wrap in cached streamlit function"

**What was wrong / how I fixed it:**
Used `n_init='auto'` which only works on newer sklearn versions, mine was 1.3 so it threw a TypeError. Changed to `n_init=10` with `init='k-means++'` explicitly.

---

### Prompt 10

**Component:** TF-IDF keyword extraction for chatbot suggestions (`app.py`)

**Prompt:**
> "extract top keywords from a list of strings using tfidf sklearn"

**What was wrong / how I fixed it:**
Basic extraction worked fine but when I tried to display keywords as buttons in a loop all buttons triggered the same last keyword (classic python closure bug). AI didn't warn about this. Fixed it by using a named callback function with `args=(kw,)` passed to `on_click` instead of a lambda.

---

## Part 4 — AI Integration & Deployment

Last part. Added the Groq API fallback + local Ollama setup, then automated AWS deployment.

---

### Prompt 11

**Component:** Groq API + Ollama fallback function (`app.py`)

**Prompt:**
> "python function that calls groq api and falls back to ollama local api if groq key not set, both return string"

**What was wrong / how I fixed it:**
AI put both providers inside one big try-except which meant an import error on groq would silently skip ollama too. I split it into independent code paths — a module-level `_GROQ_AVAILABLE` flag set at import, then two separate if-blocks so each provider fails independently.

---

### Prompt 12

**Component:** AI summary button helper (`app.py`)

**Prompt:**
> "streamlit button that shows a spinner then displays result text below the chart, make it reusable as a function"

**What was wrong / how I fixed it:**
The generated function used `st.button()` and `st.spinner()` directly using global streamlit context — this caused buttons from different tabs to appear in the wrong place. I passed the `tab` object explicitly as a parameter and called `tab.button()`, `tab.info()` on it instead.

---

### Prompt 13

**Component:** AWS deploy script (`deploy.sh`)

**Prompt:**
> "bash script for ubuntu ec2 install python3 venv install requirements and start streamlit in background"

**What was wrong / how I fixed it:**
Used `nohup streamlit run app.py &` with no log redirection so all output was lost and process died when SSH session closed. Added `> streamlit.log 2>&1` for logging and added notes in the deploy guide about using tmux or systemd for production persistence.

---
