import streamlit as st
st.set_page_config(page_title="CineAI Ultra", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import ast, requests, json, os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from thefuzz import process
import plotly.graph_objects as go
import plotly.express as px

# ── CONFIG ─────────────────────────────────────────────────────
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_IMG     = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH  = "https://api.themoviedb.org/3/search/movie"
PROFILE_FILE = "user_profile.json"

MOODS = {
    "Happy":      {"emoji":"😄","desc":"Light, fun, feel-good films",    "genres":["Comedy","Animation","Family"],       "min_rating":6.5,"keywords":["fun","comedy","laugh","joy","adventure"]},
    "Sad":        {"emoji":"😭","desc":"Emotional, touching stories",     "genres":["Drama","Romance"],                   "min_rating":7.0,"keywords":["loss","love","emotion","grief","beautiful"]},
    "Dark":       {"emoji":"💀","desc":"Intense, dark, psychological",    "genres":["Thriller","Horror","Crime"],         "min_rating":6.5,"keywords":["dark","crime","murder","thriller","mystery"]},
    "Mind-Blown": {"emoji":"🧠","desc":"Thought-provoking, complex",      "genres":["Science Fiction","Mystery"],         "min_rating":7.5,"keywords":["mind","reality","future","space","time","twist"]},
    "Excited":    {"emoji":"⚡","desc":"Action, adrenaline, epic",        "genres":["Action","Adventure"],                "min_rating":6.5,"keywords":["action","hero","battle","war","fight","epic"]},
    "Romantic":   {"emoji":"💕","desc":"Love stories, warmth",            "genres":["Romance","Drama"],                   "min_rating":6.5,"keywords":["love","romance","relationship","heart","couple"]},
}

# ── SESSION STATE ──────────────────────────────────────────────
if 'history'      not in st.session_state: st.session_state.history      = []
if 'liked'        not in st.session_state: st.session_state.liked        = []
if 'disliked'     not in st.session_state: st.session_state.disliked     = []
if 'genre_counts' not in st.session_state: st.session_state.genre_counts = {}

def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE) as f:
            d = json.load(f)
            st.session_state.history      = d.get('history', [])
            st.session_state.liked        = d.get('liked', [])
            st.session_state.disliked     = d.get('disliked', [])
            st.session_state.genre_counts = d.get('genre_counts', {})

def save_profile():
    with open(PROFILE_FILE, 'w') as f:
        json.dump({'history': st.session_state.history, 'liked': st.session_state.liked,
                   'disliked': st.session_state.disliked, 'genre_counts': st.session_state.genre_counts}, f)

def add_watch(title, df):
    row = df[df['title'] == title]
    genres = row['genres_clean'].values[0] if len(row) else ''
    st.session_state.history = [x for x in st.session_state.history if x['title'] != title]
    st.session_state.history.insert(0, {'title': title, 'genres': genres, 'time': datetime.now().strftime("%b %d, %H:%M")})
    for g in genres.split():
        st.session_state.genre_counts[g] = st.session_state.genre_counts.get(g, 0) + 1
    save_profile()

def do_feedback(title, kind, df):
    if kind == 'like':
        if title not in st.session_state.liked:   st.session_state.liked.append(title)
        if title in st.session_state.disliked:     st.session_state.disliked.remove(title)
        row = df[df['title'] == title]
        if len(row):
            for g in row['genres_clean'].values[0].split():
                st.session_state.genre_counts[g] = st.session_state.genre_counts.get(g, 0) + 2
    else:
        if title not in st.session_state.disliked: st.session_state.disliked.append(title)
        if title in st.session_state.liked:         st.session_state.liked.remove(title)
    save_profile()

def top_genre():
    gc = st.session_state.genre_counts
    if not gc: return "—"
    return max(gc, key=gc.get)

def genre_breakdown():
    gc = st.session_state.genre_counts
    if not gc: return {}
    top = sorted(gc.items(), key=lambda x: x[1], reverse=True)[:8]
    total = sum(v for _, v in top)
    return {k: round(v/total*100, 1) for k, v in top}

load_profile()

# ── CSS ───────────────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv").head(2000)
    cols = ['title','overview','genres','keywords','vote_average','release_date','popularity','vote_count']
    df = df[cols].dropna()
    def names(col):
        try: return " ".join([i['name'] for i in ast.literal_eval(col)])
        except: return ""
    df['genres_clean']   = df['genres'].apply(names)
    df['keywords_clean'] = df['keywords'].apply(names)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    df['soup'] = df['overview'] + " " + df['genres_clean'] + " " + df['keywords_clean']
    return df.reset_index(drop=True)

@st.cache_data
def build_tfidf_sim(df):
    tfidf  = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2))
    matrix = tfidf.fit_transform(df['soup'])
    return cosine_similarity(matrix[:2000], matrix[:2000])

@st.cache_data
def build_collab_sim(df):
    scaler   = MinMaxScaler()
    features = scaler.fit_transform(df[['vote_average','popularity','vote_count']].fillna(0))
    return cosine_similarity(features, features)

def hybrid_score(title, df, csim, psim, weights, n=12):
    indices = pd.Series(df.index, index=df['title'])
    if title not in indices: return pd.DataFrame()
    idx      = indices[title]
    scaler   = MinMaxScaler()
    max_year = df['year'].max()
    content_s   = csim[idx]
    pop_s       = scaler.fit_transform(df[['popularity']].fillna(0)).flatten()
    rating_s    = scaler.fit_transform(df[['vote_average']].fillna(0)).flatten()
    recency_s   = np.where(df['year'] > 0, (df['year'] - df['year'].min()) / (max_year - df['year'].min() + 1), 0)
    input_genre = df.iloc[idx]['genres_clean']
    diversity_s = np.array([0.3 if g == input_genre else 1.0 for g in df['genres_clean']])
    w = weights
    total = w['content']+w['pop']+w['rating']+w['recency']+w['diversity']
    final = (w['content']/total*content_s + w['pop']/total*pop_s + w['rating']/total*rating_s +
             w['recency']/total*recency_s + w['diversity']/total*diversity_s)
    final[idx] = 0
    result = df.copy()
    result['final_score'] = final
    result['content_s']   = content_s
    result['pop_s']       = pop_s
    result['rating_s']    = rating_s
    result['recency_s']   = recency_s
    result['diversity_s'] = diversity_s
    return result.nlargest(n, 'final_score')

def fuzzy_match(query, titles, limit=8):
    results = process.extract(query, titles, limit=limit)
    return [r[0] for r in results if r[1] > 45]

@st.cache_data(ttl=3600)
def get_poster(title, year=""):
    try:
        r = requests.get(TMDB_SEARCH, params={"api_key": TMDB_API_KEY, "query": title, "year": year}, timeout=4)
        res = r.json().get("results", [])
        if res and res[0].get("poster_path"):
            return TMDB_IMG + res[0]["poster_path"]
    except: pass
    return None

def mood_filter(df, mood_data, min_rating=6.0, n=9):
    filtered = df[df['vote_average'] >= max(min_rating, mood_data['min_rating'])].copy()
    def mood_score(row):
        score = 0
        g = str(row['genres_clean']).lower()
        k = str(row['keywords_clean']).lower()
        o = str(row['overview']).lower()
        for genre in mood_data['genres']:
            if genre.lower() in g: score += 3
        for kw in mood_data['keywords']:
            if kw in k or kw in o: score += 1
        return score
    filtered['mood_score'] = filtered.apply(mood_score, axis=1)
    return filtered[filtered['mood_score'] > 0].nlargest(n, 'mood_score')

# ── LOAD ──────────────────────────────────────────────────────
df      = load_data()
csim    = build_tfidf_sim(df)
psim    = build_collab_sim(df)
titles  = df['title'].tolist()
total   = len(df)
avg_rat = round(df['vote_average'].mean(), 1)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-name">🧠 CineAI Ultra</div>
        <div class="sb-tag">Intelligent Discovery · v5.0</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation", ["🔍 Discover","😶‍🌫️ Mood Pick","👤 My Profile","📊 Analytics","🔥 Trending"],
                    label_visibility="collapsed")

    st.markdown("<div class='sb-sec'>Smart Filters</div>", unsafe_allow_html=True)
    min_rat  = st.slider("Min Rating",  0.0, 10.0, 6.0, 0.5)
    yr_range = st.slider("Year Range",  1950, 2017, (2000, 2017))
    min_vot  = st.slider("Min Votes",   0, 5000, 300, 100)

    st.markdown("<div class='sb-sec'>Algorithm Weights</div>", unsafe_allow_html=True)
    w_content = st.slider("Content Sim",   0.0, 1.0, 0.40, 0.05)
    w_pop     = st.slider("Popularity",    0.0, 1.0, 0.20, 0.05)
    w_rating  = st.slider("Rating",        0.0, 1.0, 0.20, 0.05)
    w_recency = st.slider("Recency Boost", 0.0, 1.0, 0.10, 0.05)
    w_div     = st.slider("Diversity",     0.0, 1.0, 0.10, 0.05)

    st.markdown(f"""
    <div class='sb-sec'>Database Stats</div>
    <div class='sb-stat'><div class='sn'>{total:,}</div><div class='sl'>Movies</div></div>
    <div class='sb-stat'><div class='sn'>{avg_rat}</div><div class='sl'>Avg Rating</div></div>
    <div class='sb-stat'><div class='sn'>{len(st.session_state.history)}</div><div class='sl'>Watched</div></div>
    """, unsafe_allow_html=True)

weights = dict(content=w_content, pop=w_pop, rating=w_rating, recency=w_recency, diversity=w_div)

# ══════════════════════════════════════════════════════════════
# DISCOVER
# ══════════════════════════════════════════════════════════════
if "Discover" in page:
    st.markdown(f"""
    <div class="topbar">
        <div>
            <div class="pg-title">Movie Discovery Engine</div>
            <div class="pg-sub">Multi-signal AI · {total:,} films · Personalized for you</div>
        </div>
        <div class="live-badge"><div class="pulse"></div> ENGINE LIVE</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics">
        <div class="mc"><div class="mi">🎬</div><div class="mv">{total:,}</div><div class="ml">Films Indexed</div><div class="mb">↑ TMDB Dataset</div></div>
        <div class="mc"><div class="mi">🧠</div><div class="mv">5-Signal</div><div class="ml">Algorithm</div><div class="mb">↑ Research Grade</div></div>
        <div class="mc"><div class="mi">👤</div><div class="mv">{len(st.session_state.history)}</div><div class="ml">Watched</div><div class="mb">↑ Building Profile</div></div>
        <div class="mc"><div class="mi">🎭</div><div class="mv">{top_genre()}</div><div class="ml">Top Genre</div><div class="mb">↑ From History</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="search-box"><div class="sec-title">🔍 Search Any Movie</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([4,1,1])
    with c1:
        query = st.text_input("Movie", placeholder="Type: Inception, Dark Knight, Avatar…", label_visibility="collapsed")
    with c2:
        n_res = st.slider("Results", 3, 12, 6)
    with c3:
        st.write(""); discover_clicked = st.button("✦ Discover")
    st.markdown('</div>', unsafe_allow_html=True)

    selected = None
    if query:
        matches = fuzzy_match(query, titles)
        if matches: selected = st.selectbox("Best matches:", matches)
        else: st.warning("No match found. Try a different spelling.")

    if st.session_state.history and not query:
        last = st.session_state.history[0]['title']
        st.info(f"🎯 Smart pick based on last watch: **{last}**. Hit Discover!")
        selected = last

    if discover_clicked and selected:
        add_watch(selected, df)
        with st.spinner("🧠 Computing recommendations..."):
            results = hybrid_score(selected, df, csim, psim, weights, n_res * 2)

        if results is None or results.empty:
            st.error("Could not generate recommendations.")
        else:
            results['year'] = pd.to_datetime(results['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
            filtered = results[
                (results['vote_average'] >= min_rat) &
                (results['year'] >= yr_range[0]) &
                (results['year'] <= yr_range[1]) &
                (results['vote_count'] >= min_vot)
            ].head(n_res)

            if len(filtered) > 0 and 'content_s' in filtered.columns:
                top5 = filtered.head(5)
                fig = go.Figure()
                for comp, col, lab in zip(
                    ['content_s','pop_s','rating_s','recency_s','diversity_s'],
                    ['#7fff4f','#4fd94f','#2db82d','#1a8c1a','#0d660d'],
                    ['Content','Popularity','Rating','Recency','Diversity']
                ):
                    if comp in top5.columns:
                        fig.add_trace(go.Bar(name=lab, x=top5['title'].str[:18], y=top5[comp], marker_color=col))
                fig.update_layout(barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#e6f0e6', family='Outfit'), height=200,
                                  margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation='h', y=1.15),
                                  xaxis=dict(gridcolor='rgba(127,255,79,0.08)'),
                                  yaxis=dict(gridcolor='rgba(127,255,79,0.08)'))
                st.markdown('<div class="sec-title">📊 Score Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="results-hdr">
                <div class="sec-title" style="margin-bottom:0">🎯 Because you liked <em>{selected}</em></div>
                <div class="rc">Showing <strong>{len(filtered)}</strong> results</div>
            </div>""", unsafe_allow_html=True)

            if filtered.empty:
                st.warning("No results after filtering. Relax sidebar filters.")
            else:
                cols = st.columns(3)
                for i, (_, row) in enumerate(filtered.iterrows()):
                    year     = str(int(row['year'])) if row['year'] else 'N/A'
                    rat      = float(row['vote_average'])
                    fill     = int(rat * 10)
                    genre    = row['genres_clean'].split()[0] if row['genres_clean'] else 'Film'
                    pop      = round(float(row['popularity']), 1)
                    overview = str(row['overview'])[:150] + "…" if len(str(row['overview'])) > 150 else str(row['overview'])
                    score    = round(float(row.get('final_score', 0)) * 100, 1)
                    poster   = get_poster(row['title'], year)
                    poster_html = f'<img src="{poster}" loading="lazy"/>' if poster else '<div class="no-poster">🎬</div>'
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="mcard">
                            <div class="pw">{poster_html}
                                <div class="pr">#{i+1}</div>
                                <div class="prat">⭐ {rat}</div>
                                <div class="pscore">AI {score}%</div>
                            </div>
                            <div class="cb">
                                <div class="cgp">{genre}</div>
                                <div class="ctitle">{row['title']}</div>
                                <div class="cov">{overview}</div>
                                <div class="rbar"><div class="rf" style="width:{fill}%"></div></div>
                                <div class="cfooter"><span class="cy">📅 {year}</span><span class="cp">🔥 {pop}</span></div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        fb1, fb2 = st.columns(2)
                        with fb1:
                            if st.button("👍 Like", key=f"like_{i}"):
                                do_feedback(row['title'], 'like', df); st.toast(f"👍 Liked!")
                        with fb2:
                            if st.button("👎 Skip", key=f"skip_{i}"):
                                do_feedback(row['title'], 'dislike', df); st.toast("👎 Noted!")

# ══════════════════════════════════════════════════════════════
# MOOD
# ══════════════════════════════════════════════════════════════
elif "Mood" in page:
    st.markdown("""
    <div class="pg-title" style="padding:28px 0 8px">😶‍🌫️ Mood-Based Discovery</div>
    <div class="pg-sub" style="margin-bottom:28px">Tell us how you feel → we find the perfect film</div>
    """, unsafe_allow_html=True)

    mood_cols = st.columns(len(MOODS))
    for i, (mood_name, mood_data) in enumerate(MOODS.items()):
        with mood_cols[i]:
            if st.button(f"{mood_data['emoji']} {mood_name}", key=f"mood_{mood_name}", use_container_width=True):
                st.session_state['mood'] = mood_name

    if 'mood' in st.session_state:
        m = st.session_state['mood']
        md = MOODS[m]
        st.markdown(f"""
        <div class="mood-banner">
            <div class="mood-emoji">{md['emoji']}</div>
            <div><div class="mood-title">You're feeling {m}</div><div class="mood-desc">{md['desc']}</div></div>
        </div>""", unsafe_allow_html=True)
        mood_results = mood_filter(df, md, min_rat, n=9)
        st.markdown(f'<div class="sec-title">🎬 Perfect for your mood · {len(mood_results)} films</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (_, row) in enumerate(mood_results.iterrows()):
            year  = str(int(row['year'])) if row.get('year') else 'N/A'
            rat   = float(row['vote_average'])
            fill  = int(rat * 10)
            genre = row['genres_clean'].split()[0] if row['genres_clean'] else 'Film'
            poster = get_poster(row['title'], year)
            poster_html = f'<img src="{poster}" loading="lazy"/>' if poster else '<div class="no-poster">🎬</div>'
            with cols[i % 3]:
                st.markdown(f"""
                <div class="mcard">
                    <div class="pw">{poster_html}<div class="pr">#{i+1}</div><div class="prat">⭐ {rat}</div></div>
                    <div class="cb">
                        <div class="cgp">{genre}</div>
                        <div class="ctitle">{row['title']}</div>
                        <div class="cov">{str(row['overview'])[:130]}…</div>
                        <div class="rbar"><div class="rf" style="width:{fill}%"></div></div>
                        <div class="cfooter"><span class="cy">📅 {year}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PROFILE
# ══════════════════════════════════════════════════════════════
elif "Profile" in page:
    st.markdown('<div class="pg-title" style="padding:28px 0 18px">👤 Your Taste Profile</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("🎬 Search movies to build your taste profile!")
    else:
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown('<div class="sec-title">🎭 Genre DNA</div>', unsafe_allow_html=True)
            gdata = genre_breakdown()
            if gdata:
                fig = px.bar(x=list(gdata.values()), y=list(gdata.keys()), orientation='h',
                             color=list(gdata.values()), color_continuous_scale=[[0,'#0d200d'],[1,'#7fff4f']])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#e6f0e6', family='Outfit'), height=260,
                                  margin=dict(l=0,r=0,t=0,b=0), showlegend=False, coloraxis_showscale=False,
                                  xaxis=dict(gridcolor='rgba(127,255,79,0.08)'), yaxis=dict(gridcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"""
            <div class="profile-stats">
                <div class="ps-item"><div class="ps-num">{len(st.session_state.history)}</div><div class="ps-lbl">Watched</div></div>
                <div class="ps-item"><div class="ps-num">{len(st.session_state.liked)}</div><div class="ps-lbl">Liked</div></div>
                <div class="ps-item"><div class="ps-num">{len(st.session_state.disliked)}</div><div class="ps-lbl">Skipped</div></div>
                <div class="ps-item"><div class="ps-num">{top_genre()}</div><div class="ps-lbl">Top Genre</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:24px">📖 Watch History</div>', unsafe_allow_html=True)
        for i, item in enumerate(st.session_state.history[:10], 1):
            s = "👍" if item['title'] in st.session_state.liked else ("👎" if item['title'] in st.session_state.disliked else "🎬")
            st.markdown(f"""
            <div class="hist-card">
                <div class="hn">{i:02d}</div>
                <div class="hbody"><div class="htitle">{s} {item['title']}</div>
                <div class="htime">{item.get('genres','')[:40]} · {item.get('time','')}</div></div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Reset Profile"):
            st.session_state.history = []; st.session_state.liked = []
            st.session_state.disliked = []; st.session_state.genre_counts = {}
            if os.path.exists(PROFILE_FILE): os.remove(PROFILE_FILE)
            st.rerun()

# ══════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div class="pg-title" style="padding:28px 0 18px">📊 System Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-title">⭐ Rating Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='vote_average', nbins=20, color_discrete_sequence=['#7fff4f'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#e6f0e6'), height=260, margin=dict(l=0,r=0,t=0,b=0),
                          xaxis=dict(gridcolor='rgba(127,255,79,0.08)'), yaxis=dict(gridcolor='rgba(127,255,79,0.08)'))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="sec-title">📅 Movies by Decade</div>', unsafe_allow_html=True)
        df2 = df.copy(); df2['decade'] = (df2['year'] // 10 * 10).astype(str) + "s"
        dc = df2['decade'].value_counts().sort_index()
        fig2 = px.bar(x=dc.index, y=dc.values, color_discrete_sequence=['#4fd94f'])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#e6f0e6'), height=260, margin=dict(l=0,r=0,t=0,b=0),
                           xaxis=dict(gridcolor='rgba(127,255,79,0.08)'), yaxis=dict(gridcolor='rgba(127,255,79,0.08)'))
        st.plotly_chart(fig2, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sec-title">🔥 Popularity vs Rating</div>', unsafe_allow_html=True)
        sample = df.sample(min(300, len(df)))
        fig3 = px.scatter(sample, x='vote_average', y='popularity', hover_name='title',
                          color='vote_average', color_continuous_scale=[[0,'#0d200d'],[1,'#7fff4f']])
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#e6f0e6'), height=260, margin=dict(l=0,r=0,t=0,b=0),
                           coloraxis_showscale=False,
                           xaxis=dict(gridcolor='rgba(127,255,79,0.08)'), yaxis=dict(gridcolor='rgba(127,255,79,0.08)'))
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        st.markdown('<div class="sec-title">🎭 Top Genres</div>', unsafe_allow_html=True)
        from collections import Counter
        all_genres = []
        for g in df['genres_clean']: all_genres.extend(str(g).split())
        gf = Counter(all_genres).most_common(8)
        fig4 = px.bar(x=[g[1] for g in gf], y=[g[0] for g in gf], orientation='h', color_discrete_sequence=['#2db82d'])
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#e6f0e6'), height=260, margin=dict(l=0,r=0,t=0,b=0),
                           xaxis=dict(gridcolor='rgba(127,255,79,0.08)'), yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig4, use_container_width=True)
    st.markdown('<div class="sec-title">🏆 Top 10 Highest Rated</div>', unsafe_allow_html=True)
    top10 = df[df['vote_count'] > 500].nlargest(10, 'vote_average')[['title','vote_average','year','genres_clean','popularity','vote_count']]
    top10.columns = ['Title','Rating','Year','Genres','Popularity','Votes']
    st.dataframe(top10.style.background_gradient(cmap='Greens', subset=['Rating']), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TRENDING
# ══════════════════════════════════════════════════════════════
elif "Trending" in page:
    st.markdown('<div class="pg-title" style="padding:28px 0 18px">🔥 Trending & Popular</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔥 Most Popular", "⭐ Highest Rated", "🆕 Most Recent"])

    def render_cards(subset):
        cols = st.columns(3)
        for i, (_, row) in enumerate(subset.iterrows()):
            year = str(int(row['year'])) if row['year'] else 'N/A'
            rat  = float(row['vote_average']); fill = int(rat * 10)
            poster = get_poster(row['title'], year)
            poster_html = f'<img src="{poster}" loading="lazy"/>' if poster else '<div class="no-poster">🎬</div>'
            with cols[i % 3]:
                st.markdown(f"""
                <div class="mcard">
                    <div class="pw">{poster_html}<div class="pr">#{i+1}</div><div class="prat">⭐ {rat}</div></div>
                    <div class="cb">
                        <div class="ctitle">{row['title']}</div>
                        <div class="rbar"><div class="rf" style="width:{fill}%"></div></div>
                        <div class="cfooter"><span class="cy">📅 {year}</span><span class="cp">🔥 {round(row['popularity'],0)}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with tab1: render_cards(df.nlargest(9, 'popularity'))
    with tab2: render_cards(df[df['vote_count'] > 1000].nlargest(9, 'vote_average'))
    with tab3: render_cards(df[df['year'] >= 2010].nlargest(9, 'year'))