# 🎵 Song Similarity Recommender
### Final Project — AI/ML Decision Tool | Applied Machine Learning | Villanova University

A Streamlit web app that recommends similar songs based on audio features from the Spotify Tracks Dataset. Type in any song name and instantly get the top matches with album art, similarity scores, and brief explanations of *why* they sound alike.

---

## 🚀 Live Demo

> **[Click here to try the app →](https://your-app-name.streamlit.app)**
> *(Replace this link with your Streamlit Cloud URL after deployment)*

---

## 🧠 How It Works

| Component | What We Built |
|-----------|---------------|
| **A. ML Model** | K-Nearest Neighbors (KNN) with cosine similarity, trained on 9 normalized audio features from 600k+ Spotify tracks |
| **B. AI Component** | Music feature embeddings + intelligent explanation engine that generates unique plain-English reasoning for each recommendation |
| **C. Output/Decision** | Top N similar songs ranked by similarity %, each with album art, similarity score, and a personalized explanation |

---

## 📁 Project Structure

```
song-recommender/
├── streamlit_app.py          # Main Streamlit app
├── tracks_slim.csv           # Preprocessed Spotify dataset (slim version)
├── knn_model.pkl             # Trained KNN model
├── scaler.pkl                # MinMaxScaler for feature normalization
├── feature_matrix.pkl        # Precomputed feature vectors
├── config.json               # Column name config
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🛠️ Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/song-recommender.git
cd song-recommender
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Cloud

1. Push all files to the root of a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select repo → main file: `streamlit_app.py`
4. Click **Deploy** — live in ~2 minutes!

---

## 📊 Dataset

**Source:** [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

~600,000 tracks with audio features pre-computed by Spotify including danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, speechiness, and loudness.

---

## 🔍 Model Details

- **Algorithm:** K-Nearest Neighbors (`sklearn.neighbors.NearestNeighbors`)
- **Distance metric:** Cosine similarity
- **Preprocessing:** MinMax normalization scaling all features to [0, 1]

---

## ✨ App Features

- 🔍 Fuzzy search — typo-tolerant with suggestions
- 🖼️ Album art — fetched live from the iTunes API
- 🎯 Similarity score — percentage match per recommendation
- 💬 Unique AI explanations — personalized reasoning per song
- 📊 Feature comparison table — progress bars comparing input vs top match
- 🎚️ Adjustable results — slider for 3–10 recommendations

---

> Add your team members here

---

*Built with Streamlit · scikit-learn · Spotify Tracks Dataset · Album art via iTunes Search API*
