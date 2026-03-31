import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

body, .stApp {
    background-color: #0a0a0a;
    color: #f0f0f0;
}

.stApp {
    background: radial-gradient(ellipse at top left, #1a0a2e 0%, #0a0a0a 50%, #0d1a0a 100%);
}

h1 {
    font-family: 'Bebas Neue', cursive;
    font-size: 3.5rem !important;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #f5c518, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}

.subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 0.75rem;
    margin-bottom: 2rem;
}

.stSlider label, .stNumberInput label, .stSelectbox label, .stTextInput label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #888 !important;
}

div[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 4px !important;
    color: #f0f0f0 !important;
}

.stNumberInput input, .stTextInput input {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #f0f0f0 !important;
    border-radius: 4px !important;
}

.stButton button {
    background: linear-gradient(135deg, #f5c518, #ff6b35) !important;
    color: #0a0a0a !important;
    font-family: 'Bebas Neue', cursive !important;
    font-size: 1.3rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    border-radius: 4px !important;
    margin-top: 1rem !important;
}

.result-box {
    background: linear-gradient(135deg, #1a1a1a, #111);
    border: 1px solid #f5c518;
    border-radius: 8px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}

.rating-number {
    font-family: 'Bebas Neue', cursive;
    font-size: 6rem;
    background: linear-gradient(135deg, #f5c518, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.rating-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #666;
    margin-top: 0.5rem;
}

.section-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 1rem;
    letter-spacing: 3px;
    color: #f5c518;
    border-bottom: 1px solid #222;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.info-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: #555;
    letter-spacing: 1px;
    margin-top: -0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

MEDIANS = {
    "total_votes": 484.0, "votes_10": 51.0, "votes_9": 24.0, "votes_8": 49.0,
    "votes_7": 77.0, "votes_6": 76.0, "votes_5": 53.0, "votes_4": 30.0,
    "votes_3": 19.0, "votes_2": 14.0, "votes_1": 25.0,
    "allgenders_0age_votes": 0.0, "allgenders_18age_votes": 42.0,
    "allgenders_30age_votes": 172.0, "allgenders_45age_votes": 123.0,
    "males_allages_votes": 308.0, "males_0age_votes": 0.0,
    "males_18age_votes": 29.0, "males_30age_votes": 140.0,
    "males_45age_votes": 104.0, "females_allages_votes": 58.0,
    "females_0age_votes": 0.0, "females_18age_votes": 9.0,
    "females_30age_votes": 26.0, "females_45age_votes": 15.0,
    "top1000_voters_votes": 29.0, "us_voters_votes": 80.0,
    "non_us_voters_votes": 225.0, "high_votes_ratio": 0.2754,
    "low_votes_ratio": 0.0944, "vote_balance": 0.1752,
    "votes": 484.0, "metascore": 0.0,
    "reviews_from_users": 8.0, "reviews_from_critics": 6.0,
    "usa_gross_income": 1309583.0, "worlwide_gross_income": 1113086.0,
    "profit": -901914.0,
    "director": 6.35, "writer": 6.35, "actor": 6.35, "production_company": 6.35,
}

GENRE_MEDIAN    = 6.35
COUNTRY_MEDIAN  = 5.75
LANGUAGE_MEDIAN = 5.85

@st.cache_resource
def load_all():
    model  = joblib.load("movie_rating_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("columns.json") as f:
        columns = json.load(f)
    with open("genre_map.json") as f:
        genre_map = json.load(f)
    with open("country_map.json") as f:
        country_map = json.load(f)
    with open("language_map.json") as f:
        language_map = json.load(f)
    with open("director_map.json") as f:
        director_map = json.load(f)
    with open("writer_map.json") as f:
        writer_map = json.load(f)
    with open("actor_map.json") as f:
        actor_map = json.load(f)
    with open("company_map.json") as f:
        company_map = json.load(f)
    return model, scaler, columns, genre_map, country_map, language_map, director_map, writer_map, actor_map, company_map

try:
    model, scaler, columns, genre_map, country_map, language_map, director_map, writer_map, actor_map, company_map = load_all()
except Exception as e:
    st.error(f"⚠️ Error loading files: {e}")
    st.stop()

def encode(val, mapping, fallback):
    v = mapping.get(val, fallback)
    if isinstance(v, str):
        return fallback
    return float(v)

def encode_text(val, mapping, fallback):
    """بيبحث عن الاسم في الـ map، لو مش لاقيه بيرجع median"""
    if not val or val.strip() == "":
        return fallback
    # exact match
    if val in mapping:
        v = mapping[val]
        return float(v) if not isinstance(v, str) else fallback
    # partial match
    val_lower = val.lower()
    for k, v in mapping.items():
        if val_lower in k.lower():
            return float(v) if not isinstance(v, str) else fallback
    return fallback

TOP_GENRES    = ["Drama", "Comedy", "Action", "Thriller", "Romance", "Horror",
                 "Documentary", "Animation", "Biography, Drama", "Crime, Drama"]
TOP_COUNTRIES = ["USA", "UK", "France", "Germany", "Italy", "Spain",
                 "Japan", "India", "Australia", "Canada"]
TOP_LANGUAGES = ["English", "French", "German", "Spanish", "Italian",
                 "Japanese", "Hindi", "Arabic", "Korean", "Portuguese"]

genre_list    = TOP_GENRES    + [g for g in sorted(genre_map.keys())    if g not in TOP_GENRES]
country_list  = TOP_COUNTRIES + [c for c in sorted(country_map.keys())  if c not in TOP_COUNTRIES]
language_list = TOP_LANGUAGES + [l for l in sorted(language_map.keys()) if l not in TOP_LANGUAGES]

st.markdown("<h1>🎬 MOVIE RATING PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict IMDb Rating Before Your Film Releases</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-title">📽 FILM INFO</p>', unsafe_allow_html=True)
    year       = st.number_input("Release Year",       min_value=1900, max_value=2030, value=2024)
    month      = st.selectbox("Release Month", list(range(1, 13)), index=5,
                               format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                       "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
    duration   = st.number_input("Duration (minutes)", min_value=1,    max_value=600,  value=110)
    budget_usd = st.number_input("Budget (USD)",       min_value=0,    value=5_000_000, step=500_000)

    st.markdown('<p class="section-title">🌍 ORIGIN</p>', unsafe_allow_html=True)
    country_sel  = st.selectbox("Country",  country_list)
    language_sel = st.selectbox("Language", language_list)
    genre_sel    = st.selectbox("Genre",    genre_list)

with col2:
    st.markdown('<p class="section-title">👥 CAST & CREW</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-note">اكتب الاسم بالإنجليزي</p>', unsafe_allow_html=True)
    director_input = st.text_input("Director", placeholder="e.g. Christopher Nolan")
    writer_input   = st.text_input("Writer",   placeholder="e.g. Quentin Tarantino")
    actor_input    = st.text_input("Lead Actor", placeholder="e.g. Tom Hanks")
    company_input  = st.text_input("Production Company", placeholder="e.g. Warner Bros.")

    st.markdown('<p class="section-title">💰 FINANCIALS</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-note">Optional — leave 0 if unknown</p>', unsafe_allow_html=True)
    usa_gross   = st.number_input("Expected USA Gross (USD)",       min_value=0, value=0, step=500_000)
    world_gross = st.number_input("Expected Worldwide Gross (USD)", min_value=0, value=0, step=1_000_000)

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("⚡ PREDICT RATING")

if predict:
    film_age    = 2024 - year
    usa_final   = usa_gross   if usa_gross   > 0 else MEDIANS["usa_gross_income"]
    world_final = world_gross if world_gross > 0 else MEDIANS["worlwide_gross_income"]
    profit_val  = world_final - budget_usd

    genre_enc    = encode(genre_sel,    genre_map,    GENRE_MEDIAN)
    country_enc  = encode(country_sel,  country_map,  COUNTRY_MEDIAN)
    language_enc = encode(language_sel, language_map, LANGUAGE_MEDIAN)

    director_enc = encode_text(director_input, director_map, MEDIANS["director"])
    writer_enc   = encode_text(writer_input,   writer_map,   MEDIANS["writer"])
    actor_enc    = encode_text(actor_input,    actor_map,    MEDIANS["actor"])
    company_enc  = encode_text(company_input,  company_map,  MEDIANS["production_company"])

    input_data = {col: MEDIANS.get(col, 0) for col in columns}
    input_data.update({
        "year": year, "month": month, "duration": duration,
        "budget_usd": budget_usd,
        "genre": genre_enc, "country": country_enc, "language": language_enc,
        "director": director_enc, "writer": writer_enc,
        "actor": actor_enc, "production_company": company_enc,
        "usa_gross_income": usa_final,
        "worlwide_gross_income": world_final,
        "profit": profit_val,
        "film_age": film_age,
    })

    input_df     = pd.DataFrame([input_data]).reindex(columns=columns, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction   = round(float(model.predict(input_scaled)[0]), 2)
    prediction   = max(1.0, min(10.0, prediction))

    if prediction >= 7.5:
        verdict, color = "🏆 HIGHLY RATED", "#f5c518"
    elif prediction >= 6.0:
        verdict, color = "👍 ABOVE AVERAGE", "#88cc44"
    elif prediction >= 5.0:
        verdict, color = "😐 AVERAGE", "#ffaa00"
    else:
        verdict, color = "👎 BELOW AVERAGE", "#ff4444"

    stars = "⭐" * int(round(prediction / 2))

    st.markdown(f"""
    <div class="result-box">
        <div class="rating-number">{prediction} / 10</div>
        <div style="font-size:1.5rem; margin:0.5rem 0;">{stars}</div>
        <div class="rating-label">PREDICTED IMDB RATING</div>
        <div style="margin-top:1rem; font-family:'Inter',sans-serif;
                    font-size:1rem; color:{color}; letter-spacing:2px;">
            {verdict}
        </div>
    </div>
    """, unsafe_allow_html=True)
