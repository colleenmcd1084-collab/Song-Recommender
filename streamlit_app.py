import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from difflib import get_close_matches

st.set_page_config(page_title='Song Recommender', page_icon='🎵', layout='centered')

CSS = ''
CSS += '<style>'
CSS += "@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&family=DM+Sans:wght@400;600&display=swap');"
CSS += "html,body,[class*=css]{font-family:'DM Sans',sans-serif;}"
CSS += ".title-block{background:linear-gradient(135deg,#38bdf8,#0369a1);padding:2.5rem 2rem;border-radius:16px;margin-bottom:2rem;text-align:center;}"
CSS += ".title-block h1{color:white;font-family:'Space Mono',monospace;font-size:2.2rem;margin:0;}"
CSS += ".title-block p{color:rgba(255,255,255,.9);margin:.5rem 0 0;font-size:1rem;}"
CSS += ".song-card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:.75rem;display:flex;gap:1rem;align-items:flex-start;}"
CSS += ".song-card:hover{border-color:#38bdf8;}"
CSS += ".album-art{width:72px;height:72px;border-radius:8px;object-fit:cover;flex-shrink:0;}"
CSS += ".album-art-placeholder{width:72px;height:72px;border-radius:8px;background:#2a2a4a;display:flex;align-items:center;justify-content:center;font-size:1.8rem;flex-shrink:0;}"
CSS += ".song-info{flex:1;}"
CSS += ".song-title{color:#38bdf8;font-weight:700;font-size:1.05rem;margin:0;}"
CSS += ".song-artist{color:#aaaacc;font-size:.88rem;margin:.15rem 0 0;}"
CSS += ".song-score{color:#ffffff;font-family:'Space Mono',monospace;font-size:.8rem;margin-top:.5rem;}"
CSS += ".explain-text{color:#ccccdd;font-size:.82rem;margin-top:.4rem;font-style:italic;border-left:2px solid #38bdf8;padding-left:.6rem;}"
CSS += ".input-card{background:linear-gradient(135deg,#38bdf8,#0369a1);border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;display:flex;gap:1rem;align-items:center;}"
CSS += ".input-card .song-title{color:#fff;font-weight:700;font-size:1.1rem;margin:0;}"
CSS += ".input-card .song-artist{color:rgba(255,255,255,.85);font-size:.9rem;margin:.15rem 0 0;}"
CSS += ".feat-table{width:100%;border-collapse:collapse;margin-top:.5rem;font-size:.82rem;}"
CSS += ".feat-table th{color:#38bdf8;text-align:left;padding:.3rem .6rem;border-bottom:1px solid #2a2a4a;}"
CSS += ".feat-table td{color:#ccccdd;padding:.3rem .6rem;border-bottom:1px solid #1a1a2e;}"
CSS += ".feat-table tr:last-child td{border-bottom:none;}"
CSS += ".bar-wrap{background:#2a2a4a;border-radius:4px;height:8px;width:100%;}"
CSS += ".bar-fill{background:#38bdf8;border-radius:4px;height:8px;}"
CSS += "</style>"
st.markdown(CSS, unsafe_allow_html=True)

# Load all files from flat directory (no subfolders)
@st.cache_resource
def load_assets():
    with open('config.json') as f: config = json.load(f)
    df = pd.read_csv('tracks_slim.csv')
    with open('knn_model.pkl','rb') as f: knn = pickle.load(f)
    with open('scaler.pkl','rb') as f: scaler = pickle.load(f)
    with open('feature_matrix.pkl','rb') as f: fm = pickle.load(f)
    return config, df, knn, scaler, fm

@st.cache_data(show_spinner=False)
def get_album_art(track_name, artist_name):
    try:
        query = f'{track_name} {artist_name}'.replace(' ', '+')
        url   = f'https://itunes.apple.com/search?term={query}&media=music&limit=1'
        res   = requests.get(url, timeout=5).json()
        if res['resultCount'] > 0:
            return res['results'][0]['artworkUrl100'].replace('100x100','300x300')
    except Exception:
        pass
    return None

def explain_similarity(inp, rec, features):
    feat_cfg = [
        ('energy',       0.15, lambda v: 'high energy' if v>0.7 else ('moderate energy' if v>0.4 else 'low energy')),
        ('danceability', 0.15, lambda v: 'very danceable' if v>0.7 else ('moderately danceable' if v>0.4 else 'not very danceable')),
        ('valence',      0.15, lambda v: 'positive/happy mood' if v>0.6 else ('neutral mood' if v>0.3 else 'dark/melancholic mood')),
        ('acousticness', 0.15, lambda v: 'strongly acoustic' if v>0.6 else ('mixed acoustic/electric' if v>0.3 else 'fully electric sound')),
        ('tempo',        20,   lambda v: f'fast pace (~{int(v)} BPM)' if v>120 else (f'mid tempo (~{int(v)} BPM)' if v>90 else f'slow tempo (~{int(v)} BPM)')),
        ('speechiness',  0.1,  lambda v: 'rap/spoken word style' if v>0.5 else ('some spoken elements' if v>0.2 else 'mostly sung')),
        ('instrumentalness', 0.15, lambda v: 'largely instrumental' if v>0.5 else None),
        ('liveness',     0.15, lambda v: 'live performance feel' if v>0.5 else None),
    ]
    diffs = {}
    for feat, thresh, _ in feat_cfg:
        if feat in features:
            diffs[feat] = abs(float(inp[feat]) - float(rec[feat]))
    sorted_feats = sorted(diffs.items(), key=lambda x: x[1])
    parts = []
    for feat, diff in sorted_feats:
        cfg = next((c for c in feat_cfg if c[0]==feat), None)
        if not cfg: continue
        _, thresh, labeler = cfg
        label = labeler(float(inp[feat]))
        if label and diff <= thresh:
            parts.append(label)
        if len(parts) >= 3: break
    if not parts:
        return 'Very similar overall audio profile.'
    if len(parts) == 1:
        return f'Both tracks share a {parts[0]} sound.'
    elif len(parts) == 2:
        return f'Both have {parts[0]} with a {parts[1]} feel.'
    else:
        return f'{parts[0].capitalize()}, {parts[1]}, and {parts[2]} — very close match.'

def feature_table_html(inp, rec, features):
    display_feats = [f for f in ['danceability','energy','valence','acousticness','speechiness'] if f in features]
    rows = ''
    for feat in display_feats:
        iv   = float(inp[feat])
        rv   = float(rec[feat])
        ipct = int(iv * 100)
        rpct = int(rv * 100)
        rows += f'<tr><td><b>{feat.capitalize()}</b></td>'
        rows += f'<td><div class="bar-wrap"><div class="bar-fill" style="width:{ipct}%"></div></div><span style="color:#aaa;font-size:.75rem">{ipct}%</span></td>'
        rows += f'<td><div class="bar-wrap"><div class="bar-fill" style="width:{rpct}%;background:#aa88ff"></div></div><span style="color:#aaa;font-size:.75rem">{rpct}%</span></td></tr>'
    return f'<table class="feat-table"><thead><tr><th>Feature</th><th style="color:#38bdf8">Input Song</th><th style="color:#aa88ff">Top Match</th></tr></thead><tbody>{rows}</tbody></table>'

config, df, knn, scaler, feature_matrix = load_assets()
track_col  = config['track_col']
artist_col = config['artist_col']
features   = config['features']

st.markdown('<div class="title-block"><h1>🎵 Song Recommender</h1><p>Discover new music from songs you already love</p></div>', unsafe_allow_html=True)

query     = st.text_input('Search for a song', placeholder='e.g. Blinding Lights   or   Shape of You - Ed Sheeran')
n_results = st.slider('Number of recommendations', 3, 10, 5)

if query:
    q     = query.lower().strip()
    exact = df[df['search_key'].str.contains(q, na=False)]
    if len(exact) == 0:
        close = get_close_matches(q, df['search_key'].tolist(), n=5, cutoff=0.4)
        if close:
            st.warning('No exact match. Did you mean one of these?')
            choice = st.selectbox('Select the correct song:', close)
            exact  = df[df['search_key'] == choice]
        else:
            st.error('No match found. Try a different spelling or add the artist name.')
            st.stop()
    input_row = exact.iloc[0]
    input_idx = exact.index[0]

    with st.spinner('Loading...'):
        input_art = get_album_art(input_row[track_col], input_row[artist_col])
    art_html = f'<img class="album-art" src="{input_art}">' if input_art else '<div class="album-art-placeholder">🎵</div>'
    st.markdown(f'<div class="input-card">{art_html}<div class="song-info"><p class="song-title">{input_row[track_col]}</p><p class="song-artist">{input_row[artist_col]}</p></div></div>', unsafe_allow_html=True)

    X           = scaler.transform([input_row[features].values.astype(float)])
    dists, idxs = knn.kneighbors(X, n_neighbors=n_results+1)

    st.subheader(f'🎶 Top {n_results} Similar Songs')
    shown     = 0
    top_match = None
    for dist, idx in zip(dists[0], idxs[0]):
        if idx == input_idx: continue
        rec     = df.iloc[idx]
        sim_pct = round((1 - dist) * 100, 1)
        expl    = explain_similarity(input_row, rec, features)
        art_url = get_album_art(rec[track_col], rec[artist_col])
        art_tag = f'<img class="album-art" src="{art_url}">' if art_url else '<div class="album-art-placeholder">🎵</div>'
        st.markdown(f'<div class="song-card">{art_tag}<div class="song-info"><p class="song-title">{rec[track_col]}</p><p class="song-artist">{rec[artist_col]}</p><p class="song-score">Similarity: {sim_pct}%</p><p class="explain-text">{expl}</p></div></div>', unsafe_allow_html=True)
        if top_match is None: top_match = rec
        shown += 1
        if shown >= n_results: break

    if top_match is not None:
        with st.expander('📊 Feature Comparison (input vs top match)'):
            st.markdown(feature_table_html(input_row, top_match, features), unsafe_allow_html=True)

st.markdown('---')
st.caption('Built with the Spotify Tracks Dataset · KNN + Cosine Similarity · Album art via iTunes · Streamlit')
