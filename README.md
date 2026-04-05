# 🔍 Reddit Misinformation & Narrative Intelligence Dashboard

## 🔗 Live Hosted Link & Demo
**Live App:** [http://3.111.168.229:8501/](http://3.111.168.229:8501/)  
*(Note: Replace the placeholder above with your actual AWS EC2 Public IPv4 address if currently deployed.)*

---

## 📌 Project Overview
A sophisticated, end-to-end Streamlit application designed to analyze Reddit data with a focus on misinformation tracking, community engagement, sentiment analysis, and dynamic topic modeling. 

The core innovation of this project lies in its integration of advanced Natural Language Processing (NLP), conceptual semantic search, and an embedded **AI Summary Assistant**. Powered by a locally hosted Large Language Model (`Llama 3.2` via `Ollama`), the dashboard seamlessly distills complex data into concise, natural language insights without relying on third-party, closed-source APIs.

---

## ✨ Features
- **Brain-like Semantic Search Engine:** Moves beyond rigid keyword matching (SQL `LIKE` operator). Uses `all-MiniLM-L6-v2` embeddings to map concepts, allowing users to search by "meaning" instead of arbitrary character strings.
- **Embedded AI Analyst Data Summaries:** A fully integrated `Llama 3.2` AI agent generating real-time natural language summaries of Plotly charts and metrics.
- **Misinformation Footprint Tracking:** Quantifies and visualizes the prevalence of misinformative content, mapping its engagement velocity across different subreddits.
- **Advanced Sentiment Timeline:** Utilizes VADER lexicon approaches to label emotional sentiment (Positive, Negative, Neutral) and maps public opinion evolutions dynamically over a specified time series.
- **Machine Learning Clustering & UMAP Mapping:** Automatically groups untagged conversational themes using `KMeans`, executing dimensionality reduction via `UMAP` to plot 384-dimensional semantic text vectors onto a brilliant 2D interactive space.
- **Context-Aware "Chatbot" Recommendations:** Dynamically evaluates the TF-IDF feature space of search queries to actively suggest related exploration topics to the user.

---

## 🛠️ Tech Stack
**Frontend / Orchestration**
- [Streamlit](https://streamlit.io/) — Interactive, rapid analytical application framework.

**Data & Mathematical Foundation**
- `pandas` & `numpy` — Fast, robust data manipulation.
- `plotly`, `matplotlib`, `wordcloud` — Rich data visualization.

**NLP & Machine Learning Engine**
- `sentence-transformers` — Generates dense text embeddings (`all-MiniLM-L6-v2`).
- `scikit-learn` — Cosine similarity ranking, KMeans clustering, and TF-IDF extraction.
- `umap-learn` — High-dimensional mapping and reduction.
- `nltk.sentiment` — VADER lexicon-based polarity scoring.

**Generative AI Integration**
- `ollama` — Run open-source language models locally on bare metal.
- `Llama-3.2` — 8-billion parameter instruction-tuned LLM handling chart summaries.

---

## 📂 Project Structure
```text
.
├── app.py                      # Main Streamlit dashboard application
├── clean.csv                   # Preprocessed Reddit dataset (ingested by app)
├── requirements.txt            # Python package dependencies
├── deploy.sh                   # Bash script for native AWS EC2 automation
├── AWS_DEPLOY_GUIDE.md         # Step-by-step instructions for AWS deployment
├── README.md                   # This documentation file
└── .streamlit/
    └── secrets.toml            # Optional Streamlit config variables
```

---

## 📸 Screenshots

| | |
|:---:|:---:|
| ![Screenshot 1](screenshorts/1.jpeg) | ![Screenshot 2](screenshorts/2.jpeg) |
| ![Screenshot 3](screenshorts/3.jpeg) | ![Screenshot 4](screenshorts/4.jpeg) |
| ![Screenshot 5](screenshorts/5.jpeg) | ![Screenshot 6](screenshorts/6.jpeg) |
| ![Screenshot 7](screenshorts/7.jpeg) | ![Screenshot 8](screenshorts/8.jpeg) |
| ![Screenshot 9](screenshorts/9.jpeg) | ![Screenshot 10](screenshorts/10.jpeg) |

---

## 🚀 Local Installation & Execution

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running reliably on your host machine.

### 2. Setup
Clone the repository to your desktop or cloud workspace:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

Launch a virtual environment and install the dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the required local AI Model inside Ollama:
```bash
ollama pull llama3.2
```

### 3. Run the App
Ensure the Ollama daemon is active in the background, then spin up the Streamlit interface:
```bash
streamlit run app.py
```
*The app will be accessible at `http://localhost:8501`*

---

## ☁️ Deployment (AWS Native EC2 Automation)
We have fully automated the deployment process for AWS EC2 instances that operate independently of clunky Docker networks. 

Please systematically follow the **[`AWS_DEPLOY_GUIDE.md`](AWS_DEPLOY_GUIDE.md)** for a UI walkthrough on provisioning an Ubuntu server.  Once logged into the new machine simply run our bash provisioner:

```bash
chmod +x deploy.sh
./deploy.sh
```
*Your application, LLM, dependencies, and environment will gracefully build and spin up within 5-10 minutes.*
