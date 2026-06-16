"""
6 etapas — SHAP ir personalizacija
Užduotys:
  6.1  SHAP integracija — kodėl žaidimas tinkamas
  6.2  SHAP vizualizacija — bar chart (PNG → Flask)
  6.3  User Profile Vector — paieškų istorija → personalizacija
  6.4  Paieškos istorija — DB lentelė
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Be GUI – grafikai tik išsaugomi į failus, nerodomami ekrane
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── Prisijungimas prie duomenų bazės ─────────────────────────────────────────
DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
MODEL_DIR  = os.getenv("MODEL_DIR")
# SHAP PNG failai saugomi Flask static aplanke – UI juos rodo tiesiogiai
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "static", "shap")
os.makedirs(STATIC_DIR, exist_ok=True)

engine = create_engine(DB_URL, echo=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6.1  SHAP INTEGRACIJA
# ─────────────────────────────────────────────────────────────────────────────

# Žanrai kuriuos naudosime kaip binary features (1 = žaidimas turi šį žanrą)
GENRE_LIST = [
    "Action", "Adventure", "RPG", "Strategy", "Simulation",
    "Indie", "Casual", "Sports", "Racing", "Horror",
    "Puzzle", "Shooter", "Platformer", "Fighting", "Survival"
]

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Sukuria feature matricą iš games DataFrame.
    Features: genre binary flags + price_norm + user_reviews_norm

    Kodėl šie features?
      - Žanrai (15 binary): ar žaidimas priklauso kiekvienam žanrui (0 arba 1)
      - price_norm: kaina normalizuota į [0,1], clip prie 60$ (viršutinė riba)
      - popularity: log(1+reviews)/15 – normalizuotas populiarumas (žr. 04_recommend.py)
      - positive_ratio NEPRIDEDAMAS – tai yra target (ką prognozuojame), ne feature

    np.column_stack sujungia visus vektoriaus stulpelius į vieną matricą.
    """
    features = []
    for col in GENRE_LIST:
        # str.contains() tikrina ar žanro pavadinimas yra genres eilutėje
        features.append(
            df["genres"].fillna("").str.contains(col, case=False).astype(float)
        )
    # Kaina normalizuota: $60 = 1.0, $0 = 0.0, viršijantys $60 apkirpti prie 1.0
    features.append(df["price_final"].fillna(0).clip(0, 60) / 60.0)
    # Populiarumas: log transformacija kaip user-item matricoje (04 etapas)
    features.append(np.log1p(df["user_reviews"].fillna(0)) / 15.0)

    return np.column_stack(features)  # Grąžina (n_games, 17) matricą


def get_feature_names() -> list[str]:
    """Grąžina feature pavadinimų sąrašą tokia pat tvarka kaip build_feature_matrix."""
    return GENRE_LIST + ["price", "popularity"]


def train_shap_model(df_games: pd.DataFrame):
    """
    Treniruoja GradientBoosting klasifikatorių ant žaidimų duomenų.
    Tikslas (y): ar žaidimas turi aukštą reitingą (positive_ratio >= 80%).

    Kodėl GradientBoosting, o ne neural network?
      - SHAP TreeExplainer tiksliai veikia su medžių modeliais (GBM, XGBoost, RF)
      - Neural network SHAP yra apytiksliai ir lėtesni (DeepExplainer/GradientExplainer)
      - Čia modelio tikslas yra ne prognozavimas, o SHAP reikšmių generavimas

    Grąžina: modelis, X (features matrica)
    """
    print("   🤖 Treniruojame SHAP modelį...")
    X = build_feature_matrix(df_games)
    # Binarinis tikslas: 1 = gerai įvertintas (>=80% teigiamų), 0 = blogiau
    y = (df_games["positive_ratio"].fillna(0) >= 80).astype(int)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print(f"   ✅ Modelis paruoštas. Accuracy: {model.score(X, y):.3f}")
    return model, X


def explain_game(app_id: int, model, df_games: pd.DataFrame, top_k: int = 8) -> dict:
    """
    6.1 SHAP reikšmės vienam žaidimui.

    SHAP (SHapley Additive exPlanations):
      - Paaiškina kiek kiekvienas feature "prisidėjo" prie galutinio balo
      - Teigiama SHAP reikšmė → feature didina rekomendacijos tikimybę
      - Neigiama SHAP reikšmė → feature mažina rekomendacijos tikimybę
      - Pvz: RPG=1 su SHAP=+0.15 → RPG žanras labai padeda šiam žaidimui

    TreeExplainer – tikslus ir greitas medžių modeliams (ne apytiksliai kaip KernelExplainer).

    Grąžina: {"title", "features": [{"name", "value", "shap_value"}]}
    """
    row = df_games[df_games["app_id"] == app_id]
    if row.empty:
        return {}

    X = build_feature_matrix(row)
    feature_names = get_feature_names()

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # GradientBoosting klasifikacijai shap_values gali būti list iš 2 elementų
    # (po vieną klasei) – imame klasės 1 (teigiamos klasės) reikšmes
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Klasė 1 = "gerai įvertintas žaidimas"
    else:
        sv = shap_values[0]

    # Rikiuojame pagal |SHAP| – didžiausias absoliutus poveikis pirmas
    pairs = sorted(
        zip(feature_names, X[0], sv),
        key=lambda x: abs(x[2]),
        reverse=True
    )[:top_k]

    return {
        "app_id": app_id,
        "title":  row.iloc[0]["title"],
        "features": [
            {
                "name":       n,
                "value":      round(float(v), 3),   # Normalizuota feature reikšmė
                "shap_value": round(float(s), 4)    # SHAP poveikis
            }
            for n, v, s in pairs
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.2  SHAP VIZUALIZACIJA
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap(explanation: dict, save_path: str = None) -> str:
    """
    6.2 Horizontalus bar chart SHAP reikšmėms (tamsus fonas, Flask UI stilius).

    Spalvų logika:
      - Raudona (#e74c3c): teigiamas SHAP → feature didina rekomendacijos tikimybę
      - Mėlyna  (#3498db): neigiamas SHAP → feature mažina rekomendacijos tikimybę

    axvline(0): vertikali linija per 0 – vizualiai parodo teigiamą/neigiamą pusę.
    [::-1] apverčia sąrašą kad didžiausias baras būtų viršuje (ne apačioje).

    Grąžina išsaugoto PNG failo kelią (Flask jį patiekia per /static/shap/).
    """
    if not explanation or not explanation.get("features"):
        return ""

    features = explanation["features"]
    names  = [f["name"]       for f in features]
    values = [f["shap_value"] for f in features]
    # Spalva pagal ženklą: teigiama → raudona, neigiama → mėlyna
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)  # Nulinė linija
    ax.set_xlabel("SHAP reikšmė (poveikis rekomendacijai)")
    ax.set_title(f"Kodėl rekomenduojamas: {explanation['title'][:40]}")

    # Tamsus stilius – atitinka Flask UI temą
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()

    # Failas pavadinamas pagal app_id – Flask gali grąžinti tiesiai pagal URL
    if save_path is None:
        save_path = os.path.join(STATIC_DIR, f"shap_{explanation['app_id']}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 6.3  USER PROFILE VECTOR
# ─────────────────────────────────────────────────────────────────────────────

def build_user_profile(app_ids_history: list[int]) -> np.ndarray:
    """
    6.3 User Profile Vector iš žaidimų istorijos.

    Idėja: vartotojo "profilis" = peržiūrėtų žaidimų embedding vidurkis.
    Panašu į "average user embedding" techniką rekomendacijų sistemose.

    Eksponentinis svoris (np.linspace + np.exp):
      - Senesni žaidimai gauna mažesnį svorį (e^0 ≈ 1.0)
      - Naujesni žaidimai gauna didesnį svorį (e^1 ≈ 2.72)
      - Logika: paskutiniai paieškos rezultatai labiau atspindi dabartinį interesą

    Naudoja jau išsaugotus embeddings.npy – nereikia sentence_transformers modelio.
    """
    emb_path       = os.path.join(MODEL_DIR, "embeddings.npy")
    ids_path       = os.path.join(MODEL_DIR, "app_ids.npy")
    all_embeddings = np.load(emb_path).astype("float32")
    all_ids        = np.load(ids_path)

    # Žodynas app_id → indeksas embeddings masyve
    id_to_idx = {int(aid): i for i, aid in enumerate(all_ids)}

    # Randame tik tuos app_ids kurie yra embeddings masyve
    indices = [id_to_idx[aid] for aid in app_ids_history if aid in id_to_idx]
    if not indices:
        return np.zeros(all_embeddings.shape[1])  # Tuščias profilis = nulinis vektorius

    selected = all_embeddings[indices]

    # Eksponentinis svoris: linspace(0,1) → exp → normalizavimas
    # Pvz. 3 žaidimai: svoriai ≈ [1.0, 1.65, 2.72] → normalizuoti → [0.21, 0.35, 0.57]
    weights  = np.exp(np.linspace(0, 1, len(indices)))
    weights /= weights.sum()  # Normalizuojame kad suma = 1

    profile = np.average(selected, axis=0, weights=weights)
    return profile.astype("float32")


def personalized_search(app_ids_history: list[int], top_k: int = 10) -> list[dict]:
    """
    Personalizuota paieška pagal vartotojo žaidimų istoriją.

    Skirtumas nuo įprastos paieškos (03_search.py):
      - Čia naudojame vartotojo profilio vektorių (istorijos vidurkį)
      - Ten naudojame užklausos tekstą koduojant sentence-transformer modeliu
      - Čia nereikia sentence_transformers ar FAISS – tiesioginis cosine_similarity

    70/30 svoris (semantic + popularity) – ta pati logika kaip 03 ir 05 etapuose.
    """
    if not app_ids_history:
        return []

    emb_path       = os.path.join(MODEL_DIR, "embeddings.npy")
    ids_path       = os.path.join(MODEL_DIR, "app_ids.npy")
    all_embeddings = np.load(emb_path).astype("float32")
    all_ids        = np.load(ids_path)

    # Gauname profilio vektorių ir persformuojame į (1, dim) sklearn reikmėms
    profile_vec = build_user_profile(app_ids_history).reshape(1, -1)

    # cosine_similarity grąžina (1, n_games) matricą – [0] ištraukia vektorių
    sims = cosine_similarity(profile_vec, all_embeddings)[0]

    # Pašaliname žaidimus iš istorijos – nesiūlome jau matytų
    history_set = set(app_ids_history)
    id_to_idx   = {int(aid): i for i, aid in enumerate(all_ids)}
    for aid in history_set:
        if aid in id_to_idx:
            sims[id_to_idx[aid]] = -1  # -1 garantuoja kad nepateks į top_k

    # Paimame top_k*2 su rezervu (kai kurie gali nebūti DB)
    top_idx   = sims.argsort()[::-1][:top_k * 2]
    found_ids = [int(all_ids[i]) for i in top_idx]

    with engine.connect() as conn:
        placeholders = ",".join(str(i) for i in found_ids)
        rows = conn.execute(text(f"""
            SELECT app_id, title, genres, positive_ratio, price_final
            FROM games WHERE app_id IN ({placeholders})
        """)).fetchall()

    id_to_row = {r[0]: r for r in rows}
    results   = []
    for i, app_id in enumerate(found_ids):
        if app_id not in id_to_row:
            continue
        r                = id_to_row[app_id]
        semantic_score   = float(sims[top_idx[i]])
        popularity_score = float(r[3] or 0) / 100.0
        results.append({
            "app_id": r[0],
            "title":  r[1],
            "genres": r[2],
            "score":  round(semantic_score * 0.7 + popularity_score * 0.3, 3),
            "price":  r[4],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 6.4  PAIEŠKOS ISTORIJA DB
# ─────────────────────────────────────────────────────────────────────────────

def ensure_history_table():
    """
    Sukuria search_history lentelę jei neegzistuoja.
    CREATE TABLE IF NOT EXISTS – saugi operacija, neklaida jei lentelė jau yra.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS search_history (
                id         SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,       -- Flask session ID vartotojui identifikuoti
                query      TEXT NOT NULL,       -- Paieškos tekstas
                results    TEXT,                -- JSON: top 5 rezultatai
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))


def save_search(session_id: str, query: str, results: list[dict]):
    """
    Išsaugo paiešką į DB.
    Saugome tik top 5 rezultatus (results[:5]) – taupome vietą DB.
    ensure_ascii=False – lietuviški simboliai išsaugomi teisingai.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO search_history (session_id, query, results)
            VALUES (:sid, :q, :r)
        """), {
            "sid": session_id,
            "q":   query,
            "r":   json.dumps(results[:5], ensure_ascii=False)
        })


def get_search_history(session_id: str, limit: int = 10) -> list[str]:
    """
    Grąžina paskutines užklausas pagal session_id chronologine tvarka.

    DB grąžina DESC (naujausi pirmi), [::-1] apverčia į ASC (seniausias pirmas).
    Naudojama build_user_profile() – seniausios paieškos turi mažesnį svorį.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT query FROM search_history
            WHERE session_id = :sid
            ORDER BY created_at DESC
            LIMIT :lim
        """), {"sid": session_id, "lim": limit}).fetchall()
    return [r[0] for r in rows][::-1]  # Apverčiame į chronologinę tvarką


# ─────────────────────────────────────────────────────────────────────────────
# PALEIDIMAS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("📂 Krauname žaidimus iš DB...")
    with engine.connect() as conn:
        df_games = pd.read_sql("""
            SELECT app_id, title, genres, tags, price_final,
                   positive_ratio, user_reviews
            FROM games
        """, conn)
    print(f"   Žaidimų: {len(df_games)}")

    # 6.1 Treniruojame GradientBoosting modelį SHAP reikšmėms
    print("\n" + "="*60)
    print("6.1  SHAP INTEGRACIJA")
    print("="*60)
    model, X = train_shap_model(df_games)

    # Testuojame SHAP paaiškinimus keliems žinomiems žaidimams
    test_games = {
        "Dota 2":   570,
        "Skyrim":   489830,
        "Among Us": 945360,
    }

    explanations = {}
    for name, app_id in test_games.items():
        exp = explain_game(app_id, model, df_games)
        if exp:
            explanations[app_id] = exp
            print(f"\n🎮 {name}:")
            for f in exp["features"][:5]:
                # + reiškia teigiamą poveikį, - neigiamą
                bar = "+" if f["shap_value"] > 0 else "-"
                print(f"   {bar} {f['name']:<20} SHAP={f['shap_value']:+.4f}")

    # 6.2 Generuojame ir išsaugome SHAP bar chart PNG failus
    print("\n" + "="*60)
    print("6.2  SHAP VIZUALIZACIJA")
    print("="*60)
    for app_id, exp in explanations.items():
        path = plot_shap(exp)
        print(f"   ✅ PNG išsaugotas: {path}")

    # 6.3 Testuojame personalizuotą paiešką su fiktyvios istorijos žaidimais
    print("\n" + "="*60)
    print("6.3  USER PROFILE VECTOR")
    print("="*60)
    history = [570, 489830, 945360]  # Dota 2, Skyrim, Among Us – fiktyvus vartotojas
    profile = build_user_profile(history)
    print(f"   Profilio vektorius: shape={profile.shape}, norm={np.linalg.norm(profile):.3f}")

    print("\n   Personalizuotos rekomendacijos:")
    recs = personalized_search(history, top_k=5)
    for r in recs:
        print(f"   • {r['title']:<40} score={r['score']:.3f}")

    # 6.4 Testuojame paieškos istorijos išsaugojimą ir nuskaitymą
    print("\n" + "="*60)
    print("6.4  PAIEŠKOS ISTORIJA")
    print("="*60)
    ensure_history_table()  # Sukuriame lentelę jei neegzistuoja

    session_id = "test_session_001"
    # Išsaugome kelias paieškos užklausas kaip testo duomenis
    for q in ["dark fantasy RPG", "open world adventure", "crafting survival game"]:
        save_search(session_id, q, recs)

    retrieved = get_search_history(session_id)
    print(f"   Išsaugota {len(retrieved)} užklausų:")
    for q in retrieved:
        print(f"   • {q}")

    print("\n✅ 6 etapas baigtas!")