"""
7 etapas — Flask UI
app.py — pilna Flask aplikacija su visomis funkcijomis
Modeliai: SVD (collaborative filtering) + NCF v28 (neural collaborative filtering)
"""

# Operacinės sistemos funkcijos: keliai, aplinkos kintamieji
import os
# JSON formatui: duomenų serializavimas ir deserializavimas
import json
# Unikalių sesijos ID generavimui
import uuid
# Python objektų serializavimui į failus (modelio metaduomenys)
import pickle
# MD5 maišos funkcijai: užklausų kešavimui
import hashlib
# Matematiniai skaičiavimai su masyvais: vektoriai, matricos
import numpy as np
# Duomenų lentelės: DataFrame operacijos
import pandas as pd
# FAISS biblioteka: greita vektorių paieška (semantinei paieškai)
import faiss
# Flask komponentai: aplikacija, šablonai, užklausos, sesijos, JSON atsakymai
from flask import Flask, render_template, request, session, jsonify
# .env failo kintamųjų įkėlimas: DB slaptažodžiai, API raktai
from dotenv import load_dotenv
# SQLAlchemy: duomenų bazės ryšys ir SQL užklausos
from sqlalchemy import create_engine, text
# Sentence transformers: teksto pavertimas į vektorius (semantinė paieška)
from sentence_transformers import SentenceTransformer
# Gradient Boosting: SHAP modelio treniravimui
from sklearn.ensemble import GradientBoostingClassifier
# TF-IDF vektorizacija: KNN paieškai
from sklearn.feature_extraction.text import TfidfVectorizer
# K artimiausi kaimynai: turinio paieška
from sklearn.neighbors import NearestNeighbors
# Žanrų one-hot kodavimui
from sklearn.preprocessing import MultiLabelBinarizer
# SHAP: modelio sprendimų aiškinimas
import shap
# Matplotlib: grafikų generavimui (SHAP vizualizacija)
import matplotlib
# Agg backend: serveriniam grafikų generavimui be ekrano
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Google Gemini: RAG paaiškinimai
from google import genai
# PyTorch: neuroninis tinklas
import torch
import torch.nn as nn

# Įkeliame .env failo kintamuosius į aplinkos kintamuosius
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGŪRACIJA
# ─────────────────────────────────────────────────────────────────────────────

# Duomenų bazės URL suformuojamas iš .env kintamųjų
DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
# Aplankas su treniruotais modeliais
MODEL_DIR  = os.getenv("MODEL_DIR")
# Google Gemini API raktas
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# Aplankas SHAP grafikams (statiniai failai)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "shap")
# Sukuriame aplanką jei jo nėra
os.makedirs(STATIC_DIR, exist_ok=True)

# SQLAlchemy variklis: valdo ryšį su PostgreSQL
engine = create_engine(DB_URL, echo=False)
# Flask aplikacija
app    = Flask(__name__)
# Sesijos šifravimo raktas
app.secret_key = os.getenv("SECRET_KEY", "steam-recommender-2025")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALŪS KINTAMIEJI — lazy loading (įkeliami tik kai reikia)
# ─────────────────────────────────────────────────────────────────────────────

_st_model    = None  # Sentence transformer modelis
_faiss_index = None  # FAISS vektorių indeksas
_app_ids_np  = None  # Žaidimų app_id masyvas

def get_st_model():
    """Grąžina sentence transformer modelį, įkelia jei dar neįkeltas."""
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-mpnet-base-v2")
    return _st_model

def get_faiss():
    """Grąžina FAISS indeksą ir app_id masyvą."""
    global _faiss_index, _app_ids_np
    if _faiss_index is None:
        _faiss_index = faiss.read_index(os.path.join(MODEL_DIR, "games.index"))
        _app_ids_np  = np.load(os.path.join(MODEL_DIR, "app_ids.npy"))
    return _faiss_index, _app_ids_np

# ─────────────────────────────────────────────────────────────────────────────
# DUOMENŲ ĮKĖLIMAS — vykdomas vieną kartą paleidžiant serverį
# ─────────────────────────────────────────────────────────────────────────────

print("📂 Krauname duomenis...")
# Nuskaitome žaidimų lentelę — reikalinga rekomendacijoms ir SHAP
with engine.connect() as conn:
    df_games = pd.read_sql("""
        SELECT app_id, title, genres, tags, price_final,
               positive_ratio, user_reviews, header_image
        FROM games
    """, conn)

# SVD item faktoriai: kiekvienas žaidimas → latentinis vektorius
svd_item_factors = np.load(os.path.join(MODEL_DIR, "svd_item_factors.npy"))
# SVD žaidimų ID sąrašas: indeksas atitinka svd_item_factors eilutę
svd_game_ids     = list(np.load(os.path.join(MODEL_DIR, "game_ids.npy")))
# Greita peržvalga: app_id → pavadinimas
id_to_title      = dict(zip(df_games["app_id"], df_games["title"]))
# Greita peržvalga: app_id → paveikslėlio URL
id_to_header     = dict(zip(df_games["app_id"], df_games["header_image"].fillna("")))
print(f"   ✅ {len(df_games)} žaidimų")

# ─────────────────────────────────────────────────────────────────────────────
# NCF v28 MODELIS — Neural Collaborative Filtering su features   
# ─────────────────────────────────────────────────────────────────────────────

class NCFv28(nn.Module):
    """
    Neural Collaborative Filtering v28.
    Įėjimas: vartotojo embedding + žaidimo embedding + features
    Išėjimas: tikimybė [0,1] kad vartotojas rekomenduos žaidimą
    """
    def __init__(self, n_users, n_items, embed_dim, n_features):
        # Inicializuojame tėvinę PyTorch klasę
        super().__init__()
        # Embedding lentelė vartotojams: n_users × embed_dim
        self.user_emb = nn.Embedding(n_users, embed_dim)
        # Embedding lentelė žaidimams: n_items × embed_dim
        self.item_emb = nn.Embedding(n_items, embed_dim)
        # MLP įėjimo dydis: du embedding + features
        input_dim = embed_dim * 2 + n_features
        # Neuroninio tinklo sluoksniai
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),  # Sumažiname dimensiją iki 128
            nn.ReLU(),                   # Netiesiškumas: neigiamos → 0
            nn.Dropout(0.2),             # 20% neuronų išjungiami (overfitting prevencija)
            nn.Linear(128, 64),          # Toliau mažiname iki 64
            nn.ReLU(),                   # Aktyvacijos funkcija
            nn.Linear(64, 1),            # Vienas išėjimas
            nn.Sigmoid()                 # Paverčia į tikimybę [0,1]
        )

    def forward(self, user_idx, item_idx, features):
        # Gauname vartotojo embedding
        u = self.user_emb(user_idx)
        # Gauname žaidimo embedding
        i = self.item_emb(item_idx)
        # Sujungiame ir paduodame į MLP
        # squeeze(dim=-1): [batch,1] → [batch], saugiau nei squeeze() kai batch=1
        return self.mlp(torch.cat([u, i, features], dim=1)).squeeze(dim=-1)

# Modelio ir metaduomenų kintamieji
ncf_v28_model    = None
ncf_v28_metadata = None

try:
    # Įkeliame metaduomenis: indeksai, features sąrašas, dydžiai
    with open(os.path.join(MODEL_DIR, "ncf_v28_final_metadata.pkl"), "rb") as f:
        ncf_v28_metadata = pickle.load(f)
    # Sukuriame modelį su tinkamais dydžiais
    ncf_v28_model = NCFv28(
        n_users=ncf_v28_metadata["n_users"],
        n_items=ncf_v28_metadata["n_items"],
        embed_dim=ncf_v28_metadata["embed_dim"],
        n_features=ncf_v28_metadata["n_features"]
    )
    # Įkeliame ištreniruotus svorius
    ncf_v28_model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "ncf_v28_final.pth"),
        map_location="cpu",   # CPU (serveris neturi GPU)
        weights_only=True     # Saugesnis įkėlimas
    ))
    # Vertinimo režimas (išjungia Dropout)
    ncf_v28_model.eval()
    print("   ✅ NCF v28 modelis pakrautas")
except Exception as e:
    print(f"   ⚠️ NCF v28: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# SHAP MODELIS — žaidimų savybių aiškinimas
# ─────────────────────────────────────────────────────────────────────────────

# Žanrų sąrašas SHAP features generavimui
GENRE_LIST = ["Action","Adventure","RPG","Strategy","Simulation",
              "Indie","Casual","Sports","Racing","Horror",
              "Puzzle","Shooter","Platformer","Fighting","Survival"]

def build_features(df):
    """Paverčia žaidimų DataFrame į skaitinių features matricą."""
    # Kiekvienam žanrui: 1 jei žaidimas turi, 0 jei ne
    feats = [df["genres"].fillna("").str.contains(g, case=False).astype(float) for g in GENRE_LIST]
    # Normalizuota kaina [0,1]
    feats.append(df["price_final"].fillna(0).clip(0, 60) / 60.0)
    # Log-transformuotas atsiliepimų skaičius
    feats.append(np.log1p(df["user_reviews"].fillna(0)) / 15.0)
    return np.column_stack(feats)

print("🤖 Treniruojame SHAP modelį...")
# Features matrica visiems žaidimams
X_all      = build_features(df_games)
# Tikslas: ar žaidimas turi >=80% teigiamų
y_all      = (df_games["positive_ratio"].fillna(0) >= 80).astype(int)
# Gradient Boosting: išmoksta kurie features lemia gerą žaidimą
shap_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
shap_model.fit(X_all, y_all)
print("   ✅ SHAP paruoštas")

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — AI paaiškinimai
# ─────────────────────────────────────────────────────────────────────────────

gemini_client = None
try:
    # Inicializuojame Gemini klientą
    gemini_client = genai.Client(api_key=GEMINI_KEY)
    print("   ✅ Gemini paruoštas")
except Exception as e:
    print(f"   ⚠️ Gemini: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# REITINGŲ KONVERSIJA
# ─────────────────────────────────────────────────────────────────────────────

# Steam tekstiniai reitingai → skaitiniai [0,1]
RATING_SCORES = {
    "Overwhelmingly Positive": 1.0,
    "Very Positive":           0.85,
    "Mostly Positive":         0.70,
    "Positive":                0.65,
    "Mixed":                   0.50,
    "Mostly Negative":         0.30,
    "Negative":                0.20,
    "Overwhelmingly Negative": 0.10,
}

# ─────────────────────────────────────────────────────────────────────────────
# SEMANTINĖ PAIEŠKA
# ─────────────────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 10, mode: str = "balanced",
           w_sem: float = 0.8, w_ratio: float = 0.1,
           w_log: float = 0.1) -> list:
    """Semantinė paieška su kešavimu ir keliais rikiavimo režimais."""
    import math
    # MD5 raktas kešavimui
    query_hash = hashlib.md5(f"{query}:{top_k}:{mode}:{w_sem}:{w_ratio}:{w_log}".encode()).hexdigest()

    # Tikriname kešą
    with engine.connect() as conn:
        cached = conn.execute(
            text("SELECT result_json FROM search_cache WHERE query_hash = :h"),
            {"h": query_hash}
        ).fetchone()
    if cached:
        return json.loads(cached[0])

    # Paverčiame užklausą į vektorių
    model        = get_st_model()
    index, aids  = get_faiss()
    vec          = model.encode([query], convert_to_numpy=True).astype("float32")
    # Normalizuojame (FAISS reikalavimas)
    faiss.normalize_L2(vec)
    # Ieškome artimiausių vektorių
    distances, indices = index.search(vec, top_k * 3)
    found_ids    = [int(aids[i]) for i in indices[0] if i != -1]

    # Gauname žaidimų detales iš DB
    with engine.connect() as conn:
        ph   = ",".join(str(i) for i in found_ids)
        rows = conn.execute(text(f"""
            SELECT app_id, title, genres, tags, price_final,
                   positive_ratio, rating, header_image, about_the_game,
                   user_reviews
            FROM games WHERE app_id IN ({ph})
        """)).fetchall()

    id_to_row = {r[0]: r for r in rows}
    results   = []
    for app_id, dist in zip(found_ids, distances[0]):
        if app_id not in id_to_row:
            continue
        r      = id_to_row[app_id]
        # Semantinis panašumas
        sem    = float(1 - dist / 2)
        # Normalizuotas teigiamų %
        pop    = float(r[5] or 0) / 100.0
        # Log-transformuotas atsiliepimų skaičius
        rev    = math.log1p(r[9] or 0) / 15.0
        # Skaitinis reitingas
        rating = RATING_SCORES.get(r[6] or "", 0.5)

        # Galutinis balas pagal režimą
        if mode == "semantic":
            score = sem
        elif mode == "popular":
            score = sem * 0.6 + pop * 0.4
        elif mode == "logarithmic":
            score = sem * 0.8 + rev * 0.1 + pop * 0.1
        elif mode == "custom":
            total = w_sem + w_ratio + w_log or 1.0
            score = (sem * w_sem + pop * w_ratio + rev * w_log) / total
        else:
            score = sem * 0.8 + pop * 0.2

        results.append({
            "app_id": r[0], "title": r[1], "genres": r[2],
            "price": r[4], "positive_ratio": r[5], "rating": r[6],
            "header_image": r[7], "about": (r[8] or "")[:200],
            "user_reviews": r[9],
            "semantic_score": round(sem, 3),
            "score": round(score, 3),
        })

    # Rikiuojame ir apribojame
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Išsaugome į kešą
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO search_cache (query_hash, query_text, result_json)
            VALUES (:h, :q, :r) ON CONFLICT (query_hash) DO NOTHING
        """), {"h": query_hash, "q": query, "r": json.dumps(results, ensure_ascii=False)})

    return results

# ─────────────────────────────────────────────────────────────────────────────
# SVD REKOMENDACIJOS
# ─────────────────────────────────────────────────────────────────────────────

def recommend_svd_fn(app_id: int, top_k: int = 10) -> list:
    """SVD rekomendacijos: kosinusinis panašumas + žanro boost."""
    # Tikriname ar žaidimas yra modelyje
    if app_id not in svd_game_ids:
        return []
    # Žaidimo pozicija matricoje
    idx      = svd_game_ids.index(app_id)
    # Žaidimo latentinis vektorius
    vec      = svd_item_factors[idx]
    # Vektorių normos
    norms    = np.linalg.norm(svd_item_factors, axis=1)
    # Kosinusinis panašumas su visais žaidimais
    sims     = svd_item_factors.dot(vec) / (norms * np.linalg.norm(vec) + 1e-9)
    # Pašaliname patį žaidimą
    sims[idx] = -1

    # Žanro boost: +20% už sutampančius žanrus
    src = df_games[df_games["app_id"] == app_id]
    if not src.empty:
        src_genres = {g.strip().lower() for g in str(src.iloc[0]["genres"] or "").split(",") if g.strip()}
        for i, gid in enumerate(svd_game_ids):
            if gid == app_id:
                continue
            row = df_games[df_games["app_id"] == gid]
            if row.empty:
                continue
            rec_genres = {g.strip().lower() for g in str(row.iloc[0]["genres"] or "").split(",") if g.strip()}
            if src_genres & rec_genres:
                sims[i] *= 1.2

    # Top_k žaidimų
    top_idx = sims.argsort()[::-1][:top_k]
    return [{
        "app_id":       svd_game_ids[i],
        "title":        id_to_title.get(svd_game_ids[i], "?"),
        "header_image": id_to_header.get(svd_game_ids[i], ""),
        "score":        round(float(sims[i]), 3),
    } for i in top_idx]

# ─────────────────────────────────────────────────────────────────────────────
# NCF v28 REKOMENDACIJOS
# ─────────────────────────────────────────────────────────────────────────────

def recommend_ncf_v28_fn(app_id: int, top_k: int = 10) -> list:
    """
    NCF v28 rekomendacijos pagal item embedding kosinusinį panašumą.
    Jei žaidimas nėra modelyje - grąžina populiariausius pagal normą.
    """
    # Tikriname ar modelis įkeltas
    if ncf_v28_model is None or ncf_v28_metadata is None:
        return []

    # Žaidimų ID → indekso žodynas
    item_to_idx = ncf_v28_metadata["item_to_idx"]
    # Apverstas žodynas: indeksas → žaidimo ID
    idx_to_game = {v: k for k, v in item_to_idx.items()}

    ncf_v28_model.eval()
    with torch.no_grad():  # Išjungiame gradientus (greičiau)
        # Visų žaidimų indeksai
        all_item_idx   = torch.arange(ncf_v28_metadata["n_items"], dtype=torch.long)
        # Visų žaidimų embeddings
        all_embeddings = ncf_v28_model.item_emb(all_item_idx).numpy()

        if app_id in item_to_idx:
            # Žaidimas modelyje: kosinusinis panašumas
            target_idx  = item_to_idx[app_id]
            target_emb  = all_embeddings[target_idx]
            norms       = np.linalg.norm(all_embeddings, axis=1)
            target_norm = np.linalg.norm(target_emb)
            sims        = all_embeddings.dot(target_emb) / (norms * target_norm + 1e-9)
            # Pašaliname patį žaidimą
            sims[target_idx] = -1
        else:
            # Žaidimas ne modelyje: populiariausi pagal embedding normą
            norms = np.linalg.norm(all_embeddings, axis=1)
            sims  = norms / (norms.max() + 1e-9)

        # Top_k žaidimų
        top_idx = sims.argsort()[::-1][:top_k]

    # Filtruojame ir grąžiname rezultatus
    return [{
        "app_id":       idx_to_game.get(int(i), 0),
        "title":        id_to_title.get(idx_to_game.get(int(i), 0), "?"),
        "header_image": id_to_header.get(idx_to_game.get(int(i), 0), ""),
        "score":        round(float(sims[i]), 3),
    } for i in top_idx if idx_to_game.get(int(i), 0) != 0]

# ─────────────────────────────────────────────────────────────────────────────
# SHAP AIŠKINIMAS
# ─────────────────────────────────────────────────────────────────────────────

def explain_game_fn(app_id: int) -> dict:
    """Apskaičiuoja SHAP reikšmes žaidimo savybėms."""
    row = df_games[df_games["app_id"] == app_id]
    if row.empty:
        return {}
    # Features vektorius
    X           = build_features(row)
    # SHAP paaiškintojas
    explainer   = shap.TreeExplainer(shap_model)
    # SHAP reikšmės
    shap_values = explainer.shap_values(X)
    # Teigiamos klasės reikšmės
    sv          = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    names       = GENRE_LIST + ["price", "popularity"]
    # Rikiuojame pagal absoliučią reikšmę (svarbiausi pirmiau)
    pairs       = sorted(zip(names, X[0], sv), key=lambda x: abs(x[2]), reverse=True)[:8]
    return {
        "app_id":   app_id,
        "title":    row.iloc[0]["title"],
        "features": [{"name": n, "value": round(float(v), 3), "shap_value": round(float(s), 4)}
                     for n, v, s in pairs]
    }

def plot_shap_fn(exp: dict) -> str:
    """Generuoja SHAP grafiką ir išsaugo kaip PNG."""
    if not exp or not exp.get("features"):
        return ""
    features = exp["features"]
    names    = [f["name"] for f in features]
    values   = [f["shap_value"] for f in features]
    # Žalia = teigiama įtaka, raudona = neigiama
    colors   = ["#4fffb0" if v > 0 else "#ff6b6b" for v in values]
    fig, ax  = plt.subplots(figsize=(7, 4))
    # Horizontalus stulpelinis grafikas
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("SHAP reikšmė", color="white")
    ax.set_title(f"Kodėl: {exp['title'][:35]}", color="white")
    # Tamsus fonas
    ax.set_facecolor("#161922")
    fig.patch.set_facecolor("#161922")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("#2a2d3a")
    plt.tight_layout()
    # Išsaugome PNG failą
    path = os.path.join(STATIC_DIR, f"shap_{exp['app_id']}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    # Atlaisvinkame atmintį
    plt.close()
    return path

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI RAG PAAIŠKINIMAS
# ─────────────────────────────────────────────────────────────────────────────

def rag_explain_fn(query: str, top_k: int = 5) -> str:
    """Gemini AI paaiškina paieškos rezultatus."""
    if not gemini_client:
        return ""
    results    = search(query, top_k=top_k)
    # Formuojame žaidimų sąrašą kaip tekstą
    games_text = "".join(
        f"{i+1}. {g['title']} (žanrai: {g['genres']}, {g['positive_ratio']}%, ${g['price'] or 0:.2f})\n"
        for i, g in enumerate(results)
    )
    # Prompt'as Gemini modeliui
    prompt = (
        f'Tu esi žaidimų ekspertas. Vartotojas ieško: "{query}"\n\n'
        f"Rasti žaidimai:\n{games_text}\n"
        f"Paaiškink kodėl šie žaidimai tinka (2-3 sakiniai), išskirk geriausią.\n"
        f"Atsakyk lietuviškai, glaustai (max 120 žodžių)."
    )
    try:
        resp = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini: {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# RAKTINIŲ ŽODŽIŲ PAIEŠKA
# ─────────────────────────────────────────────────────────────────────────────

def search_keyword(query: str, top_k: int = 10) -> list:
    """PostgreSQL ILIKE paieška pagal pavadinimą, žanrus, tags, aprašymą."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT app_id, title, genres, tags, price_final,
                   positive_ratio, rating, header_image, about_the_game,
                   user_reviews,
                   (
                     CASE WHEN title ILIKE :q THEN 3.0 ELSE 0 END +
                     CASE WHEN tags  ILIKE :q THEN 2.0 ELSE 0 END +
                     CASE WHEN genres ILIKE :q THEN 1.5 ELSE 0 END +
                     CASE WHEN about_the_game ILIKE :q THEN 1.0 ELSE 0 END
                   ) AS kw_score
            FROM games
            WHERE title ILIKE :q OR tags ILIKE :q
               OR genres ILIKE :q OR about_the_game ILIKE :q
            ORDER BY kw_score DESC, user_reviews DESC NULLS LAST
            LIMIT :k
        """), {"q": f"%{query}%", "k": top_k}).fetchall()

    results = []
    for i, r in enumerate(rows):
        results.append({
            "app_id": r[0], "title": r[1], "genres": r[2], "tags": r[3],
            "price": r[4], "positive_ratio": r[5], "rating": r[6],
            "header_image": r[7], "about": (r[8] or "")[:200],
            "user_reviews": r[9],
            "semantic_score": round(float(r[10] or 0), 3),
            "score": round(float(r[10] or 0), 3),
            "rank": i + 1,
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# KNN PAIEŠKA
# ─────────────────────────────────────────────────────────────────────────────

_knn_model = None  # KNN modelis (lazy loading)
_knn_df    = None  # Žaidimų DataFrame KNN paieškai

def _load_knn():
    """Įkelia KNN modelį ir TF-IDF vektorizatorių."""
    global _knn_model, _knn_df
    if _knn_model is not None:
        return
    print("📐 Krauname KNN + TF-IDF...")
    with engine.connect() as conn:
        _knn_df = pd.read_sql("""
            SELECT app_id, title, genres, tags, about_the_game,
                   price_final, positive_ratio, rating, header_image, user_reviews
            FROM games
        """, conn)

    def build_text(row):
        """Sujungia teksto laukus į vieną eilutę."""
        tags  = str(row["tags"] or "")
        about = str(row["about_the_game"] or "")[:300]
        # Tags kartojame du kartus (didesnis svoris paieškai)
        return " ".join(filter(None, [
            str(row["title"] or ""), str(row["genres"] or ""),
            tags, tags, about
        ]))

    _knn_df["text"] = _knn_df.apply(build_text, axis=1)
    # TF-IDF vektorizacija
    tfidf  = TfidfVectorizer(max_features=20000, stop_words="english")
    matrix = tfidf.fit_transform(_knn_df["text"])
    # KNN modelis su kosinusiniu atstumu
    _knn_model = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
    _knn_model.fit(matrix)
    # Išsaugome TF-IDF modelyje (reikia užklausoms)
    _knn_model._tfidf  = tfidf
    _knn_model._matrix = matrix
    print("   ✅ KNN paruoštas")

def search_knn(query: str, top_k: int = 10) -> list:
    """KNN paieška pagal TF-IDF vektorių."""
    _load_knn()
    # Paverčiame užklausą į TF-IDF vektorių
    q_vec = _knn_model._tfidf.transform([query])
    # Randame artimiausius kaimynus
    dists, indices = _knn_model.kneighbors(q_vec, n_neighbors=top_k)
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], dists[0])):
        row   = _knn_df.iloc[idx]
        # Panašumas: 1 - kosinusinis atstumas
        score = round(float(1 - dist), 3)
        results.append({
            "app_id": int(row["app_id"]), "title": row["title"],
            "genres": row["genres"], "tags": row["tags"],
            "price": row["price_final"], "positive_ratio": row["positive_ratio"],
            "rating": row["rating"], "header_image": row["header_image"],
            "about": (str(row["about_the_game"] or ""))[:200],
            "user_reviews": row["user_reviews"],
            "semantic_score": score, "score": score, "rank": rank + 1,
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# PAIEŠKOS ISTORIJA
# ─────────────────────────────────────────────────────────────────────────────

def ensure_history_table():
    """Sukuria paieškos istorijos lentelę jei jos nėra."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS search_history (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                query TEXT,
                results TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

# Paleidžiame su serveriu
ensure_history_table()

def save_search(sid: str, query: str, results: list):
    """Išsaugo paieškos užklausą istorijoje."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO search_history (session_id, query, results)
            VALUES (:sid, :q, :r)
        """), {"sid": sid, "q": query, "r": json.dumps(results[:5], ensure_ascii=False)})

def get_history(sid: str) -> list:
    """Grąžina paskutines unikalias užklausas."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT query FROM search_history
            WHERE session_id = :sid
            ORDER BY created_at DESC LIMIT 20
        """), {"sid": sid}).fetchall()
    # Filtruojame dublikatus
    seen = []
    for r in rows:
        if r[0] not in seen:
            seen.append(r[0])
    return seen[:10]

def get_sid():
    """Grąžina arba sukuria sesijos ID."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Pagrindinis puslapis."""
    return render_template("index.html", history=get_history(get_sid()))


@app.route("/search")
def search_route():
    """Paieškos rezultatų puslapis."""
    query = request.args.get("q", "").strip()
    mode  = request.args.get("mode", "all")
    sid   = get_sid()
    if not query:
        return render_template("index.html", error="Įveskite užklausą", history=[])

    # Pasirinktiniai svoriai
    w_sem   = float(request.args.get("w_sem",   0.8))
    w_ratio = float(request.args.get("w_ratio", 0.1))
    w_log   = float(request.args.get("w_log",   0.1))

    # Visų 7 paieškos režimų rezultatai
    all_results = {
        "keyword":     search_keyword(query, top_k=5),
        "knn":         search_knn(query, top_k=5),
        "balanced":    search(query, top_k=5, mode="balanced"),
        "semantic":    search(query, top_k=5, mode="semantic"),
        "popular":     search(query, top_k=5, mode="popular"),
        "logarithmic": search(query, top_k=5, mode="logarithmic"),
        "custom":      search(query, top_k=5, mode="custom",
                              w_sem=w_sem, w_ratio=w_ratio, w_log=w_log),
    }

    # Išsaugome istoriją
    save_search(sid, query, all_results["balanced"])

    return render_template("index.html",
                           query=query,
                           all_results=all_results,
                           results=all_results["balanced"],
                           history=get_history(sid),
                           mode=mode,
                           w_sem=w_sem, w_ratio=w_ratio, w_log=w_log)


@app.route("/api/explain")
def api_explain():
    """API: Gemini paaiškinimas."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"explanation": ""})
    explanation = rag_explain_fn(query)
    return jsonify({"explanation": explanation})


@app.route("/game/<int:app_id>")
def game_detail(app_id):
    """Žaidimo detalių puslapis."""
    # Gauname žaidimo informaciją
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT app_id, title, genres, tags, about_the_game,
                   price_final, positive_ratio, rating, header_image
            FROM games WHERE app_id = :id
        """), {"id": app_id}).fetchone()
    if not row:
        return "Žaidimas nerastas", 404

    game = {
        "app_id": row[0], "title": row[1], "genres": row[2], "tags": row[3],
        "about": row[4], "price": row[5], "positive_ratio": row[6],
        "rating": row[7], "header_image": row[8]
    }

    # SVD rekomendacijos (collaborative filtering)
    recs_svd     = recommend_svd_fn(app_id)
    # NCF v28 rekomendacijos (neuroninis tinklas)
    recs_ncf_v28 = recommend_ncf_v28_fn(app_id)
    # SHAP paaiškinimas
    exp          = explain_game_fn(app_id)
    shap_png     = None
    if exp:
        path = plot_shap_fn(exp)
        if path:
            shap_png = os.path.basename(path)

    return render_template("game.html",
                           game=game,
                           recommendations=recs_svd,
                           recommendations_ncf_v28=recs_ncf_v28,
                           shap_explanation=exp,
                           shap_png=shap_png)


@app.route("/compare")
def compare():
    """Modelių palyginimo puslapis."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT model_name, precision_k, recall_k, trained_at
            FROM model_results ORDER BY trained_at DESC LIMIT 10
        """)).fetchall()
    metrics = [{"model": r[0], "precision": r[1], "recall": r[2], "date": str(r[3])} for r in rows]
    return render_template("compare.html", metrics=metrics)


@app.route("/profile")
def profile():
    """Vartotojo profilio puslapis."""
    sid = get_sid()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT query, results, created_at
            FROM search_history
            WHERE session_id = :sid
            ORDER BY created_at DESC LIMIT 50
        """), {"sid": sid}).fetchall()

    history_full = []
    genre_counts = {}
    for r in rows:
        results = json.loads(r[1]) if r[1] else []
        # Skaičiuojame žanrų dažnį
        for game in results:
            for g in str(game.get("genres") or "").split(","):
                g = g.strip()
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        history_full.append({
            "query": r[0],
            "results": results[:3],
            "created_at": r[2].strftime("%Y-%m-%d %H:%M") if r[2] else ""
        })

    # Top 8 žanrų
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    return render_template("profile.html",
                           history=get_history(sid),
                           history_full=history_full,
                           top_genres=top_genres,
                           session_id=sid)


@app.route("/api/search")
def api_search():
    """API: paieška JSON formatu."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Query required"}), 400
    return jsonify(search(query, top_k=int(request.args.get("top_k", 10))))


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    """API: išvalo paieškos istoriją."""
    sid = get_sid()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM search_history WHERE session_id = :sid"), {"sid": sid})
    return jsonify({"ok": True})


# Paleidžiame serverį jei failas vykdomas tiesiogiai
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)