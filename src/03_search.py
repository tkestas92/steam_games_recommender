"""
3 etapas — Semantinė paieška
Užduotys:
  3.1  Embeddings generavimas (sentence-transformers)
  3.2  FAISS indeksas
  3.3  Paieškos funkcija
  3.4  Steam Web API (header image + kaina, kešavimas DB)
  3.5  TF-IDF vs Semantic palyginimas
"""

import os
import json
import hashlib
import time
import requests
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Įkeliame aplinkos kintamuosius iš .env failo
load_dotenv()

# ── Prisijungimas prie duomenų bazės ─────────────────────────────────────────
DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
DATA_DIR  = os.getenv("DATA_DIR")   # CSV failų aplankas
MODEL_DIR = os.getenv("MODEL_DIR")  # Embeddings/FAISS failų aplankas

engine = create_engine(DB_URL, echo=False)  # echo=False – nerodo SQL į konsolę


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  EMBEDDINGS GENERAVIMAS
# ─────────────────────────────────────────────────────────────────────────────

def generate_embeddings(force=False):
    """
    Nuskaito games lentelę, sukuria text lauką, generuoja 768d embeddings.
    Išsaugo: embeddings.npy ir app_ids.npy į MODEL_DIR.
    force=True — pergeneruoja net jei failai jau yra.
    """
    emb_path = os.path.join(MODEL_DIR, "embeddings.npy")
    ids_path = os.path.join(MODEL_DIR, "app_ids.npy")

    # Jei failai jau egzistuoja ir force=False – praleisti generavimą
    if not force and os.path.exists(emb_path) and os.path.exists(ids_path):
        print("✅ Embeddings jau yra, praleidžiame (naudok force=True pergeneruoti)")
        return

    print("📂 Krauname žaidimus iš DB...")
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT app_id, title, about_the_game, tags, genres FROM games", conn
        )
    print(f"   Žaidimų: {len(df)}")

    def build_text(row):
        """
        Kiekvienam žaidimui suformuojame vieną teksto eilutę embedding modeliui.
        Tags kartojami du kartus – taip modelis labiau akcentuoja žanrą/tagus
        lyginant su laisvu aprašymu.
        """
        about = str(row["about_the_game"] or "")[:300]  # Aprašymą trumpiname iki 300 simbolių
        tags  = str(row["tags"] or "")
        return " ".join(filter(None, [
            str(row["title"] or ""),
            str(row["genres"] or ""),
            tags,
            tags,   # tags du kartus — boost žanro žodžiams
            about
        ]))

    df["text"] = df.apply(build_text, axis=1)

    # Modelis all-mpnet-base-v2 generuoja 768 dimensijų vektorius
    # Vienas iš geriausių semantinės paieškos modelių pagal MTEB benchmark
    print("🤖 Krauname modelį: all-mpnet-base-v2...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("⚙️  Generuojame embeddings...")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,          # 64 tekstai vienu metu – balansas tarp greičio ir atminties
        show_progress_bar=True,
        convert_to_numpy=True   # Grąžiname kaip numpy masyvą, ne torch tensorius
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(emb_path, embeddings.astype("float32"))  # float32 – mažiau vietos, FAISS reikalauja
    np.save(ids_path, df["app_id"].values)           # Saugome app_id atitikmenį indeksui
    print(f"✅ Embeddings išsaugoti: {embeddings.shape}  →  {emb_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  FAISS INDEKSAS
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(force=False):
    """
    Sukuria IndexFlatL2 FAISS indeksą iš embeddings.npy.
    Išsaugo games.index į MODEL_DIR.

    IndexFlatL2 – tikslus L2 atstumo (Euklido) paieškos indeksas.
    Lėtesnis už ANN (Approximate Nearest Neighbor), bet 100% tikslus.
    Po normalize_L2 L2 atstumas ekvivalentus cosine panašumui.
    """
    index_path = os.path.join(MODEL_DIR, "games.index")

    if not force and os.path.exists(index_path):
        print("✅ FAISS indeksas jau yra, praleidžiame")
        return

    emb_path = os.path.join(MODEL_DIR, "embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError("embeddings.npy nerasta — pirmiau paleisk generate_embeddings()")

    print("📐 Kuriame FAISS indeksą...")
    embeddings = np.load(emb_path).astype("float32")

    # Normalizuojame vektorius į vienetinę sferą (||v|| = 1)
    # Po to L2 atstumas = 2 * (1 - cosine_similarity), todėl galima naudoti L2 indeksą
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]  # 768 – vektoriaus dimensija
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)        # Pridedame visus vektorius į indeksą

    faiss.write_index(index, index_path)
    print(f"✅ FAISS indeksas išsaugotas: {index.ntotal} vektoriai  →  {index_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  PAIEŠKOS FUNKCIJA
# ─────────────────────────────────────────────────────────────────────────────

# Globalūs kintamieji – modelis ir indeksas įkeliami tik vieną kartą (lazy loading)
_model   = None
_index   = None
_app_ids = None

def _load_search_assets():
    """Įkelia modelį ir FAISS indeksą į atmintį jei dar neįkelti."""
    global _model, _index, _app_ids
    if _model is None:
        print("🤖 Krauname modelį...")
        _model = SentenceTransformer("all-mpnet-base-v2")
    if _index is None:
        index_path = os.path.join(MODEL_DIR, "games.index")
        ids_path   = os.path.join(MODEL_DIR, "app_ids.npy")
        _index   = faiss.read_index(index_path)
        _app_ids = np.load(ids_path)


def search(query: str, top_k: int = 10, use_cache: bool = True, mode: str = "balanced") -> list[dict]:
    """
    Semantinė paieška su keičiamu rikiavimo režimu.

    Režimai (mode):
      semantic    — tik embedding panašumas (grynas semantinis)
      balanced    — semantic 80% + positive_ratio 20%  [default]
      popular     — semantic 60% + positive_ratio 40%  (labiau pasverta į populiarumą)
      logarithmic — semantic 80% + log(reviews) 10% + ratio 10%  (mažina outlier efektą)

    Kešavimas: rezultatai saugomi DB lentelėje search_cache pagal MD5 hash
    iš (query + top_k + mode) – taip tas pats paieškos variantas nekartojamas.
    """
    import math

    # Generuojame unikalų hash pagal užklausą + parametrus
    query_hash = hashlib.md5(f"{query}:{top_k}:{mode}".encode()).hexdigest()

    # Patikriname ar rezultatas jau kešuotas
    if use_cache:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT result_json FROM search_cache WHERE query_hash = :h"),
                {"h": query_hash}
            ).fetchone()
        if row:
            print(f"💾 Cache hit: '{query}'")
            return json.loads(row[0])

    # Įkeliame modelį ir indeksą (jei dar neįkelti)
    _load_search_assets()

    # Koduojame užklausą į 768d vektorių ir normalizuojame (kaip indekso vektoriai)
    vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)

    # Ieškome top_k*3 artimiausių vektorių – imame daugiau nes kai kurie gali nebūti DB
    distances, indices = _index.search(vec, top_k * 3)
    found_ids = [int(_app_ids[i]) for i in indices[0] if i != -1]  # -1 = nerastas

    # Gauname žaidimų duomenis iš DB pagal rastus app_id
    with engine.connect() as conn:
        placeholders = ",".join(str(i) for i in found_ids)
        rows = conn.execute(text(f"""
            SELECT app_id, title, genres, tags, price_final,
                   positive_ratio, rating, header_image, about_the_game,
                   user_reviews
            FROM games
            WHERE app_id IN ({placeholders})
        """)).fetchall()

    # Žodynas app_id → eilutė greičiau paieškai nei iteravimas per sąrašą
    id_to_row = {r[0]: r for r in rows}

    results = []
    for app_id, dist in zip(found_ids, distances[0]):
        if app_id not in id_to_row:
            continue  # Žaidimas ištrintas iš DB arba filtruotas
        r = id_to_row[app_id]

        # dist yra L2 atstumas ∈ [0, 2] (normalizuotiems vektoriams)
        # Konvertuojame į panašumo balą ∈ [0, 1]: 0 = visiškai skirtingi, 1 = identiški
        semantic_score   = float(1 - dist / 2)

        # Positive ratio normalizuojamas į [0, 1] (DB saugomas kaip 0–100)
        popularity_score = float(r[5] or 0) / 100.0

        # log(1+reviews) / 15 normalizuoja apžvalgų skaičių į [0, 1] apytiksliai
        # log1p išvengiam log(0) klaidos; /15 nes log(e^15) ≈ maks tikėtinas skaičius
        reviews_log      = math.log1p(r[9] or 0) / 15.0

        # Galutinis balas pagal pasirinktą režimą
        if mode == "semantic":
            final_score = semantic_score
        elif mode == "popular":
            final_score = semantic_score * 0.6 + popularity_score * 0.4
        elif mode == "logarithmic":
            final_score = semantic_score * 0.8 + reviews_log * 0.1 + popularity_score * 0.1
        else:  # balanced (default)
            final_score = semantic_score * 0.8 + popularity_score * 0.2

        results.append({
            "app_id":         r[0],
            "title":          r[1],
            "genres":         r[2],
            "tags":           r[3],
            "price":          r[4],
            "positive_ratio": r[5],
            "rating":         r[6],
            "header_image":   r[7],
            "about":          (r[8] or "")[:200],  # Aprašymą trumpiname UI reikmėms
            "user_reviews":   r[9],
            "semantic_score": round(semantic_score, 3),
            "score":          round(final_score, 3),
            "mode":           mode,
        })

    # Rikiuojame pagal galutinį balą mažėjančia tvarka
    results.sort(key=lambda x: x["score"], reverse=True)

    # Pridedame rango numerį (1-based)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    # Paimame tik top_k rezultatų (prieš tai ieškojome top_k*3 su rezervu)
    results = results[:top_k]

    # Išsaugome rezultatą į kešą – ON CONFLICT DO NOTHING: neklaida jei jau yra
    result_json = json.dumps(results, ensure_ascii=False)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO search_cache (query_hash, query_text, result_json)
            VALUES (:h, :q, :r)
            ON CONFLICT (query_hash) DO NOTHING
        """), {"h": query_hash, "q": query, "r": result_json})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  STEAM WEB API
# ─────────────────────────────────────────────────────────────────────────────

STEAM_API_URL = "https://store.steampowered.com/api/appdetails"

def enrich_with_steam_api(app_id: int, force_refresh: bool = False) -> dict | None:
    """
    Praturtina žaidimo įrašą iš Steam Web API: header_image + kaina.
    Jei header_image jau yra DB ir force_refresh=False – naudojame kešą, nekviečiame API.
    """
    # Pirma patikriname DB kešą – taupome API kvietimus
    if not force_refresh:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT header_image FROM games WHERE app_id = :id"),
                {"id": app_id}
            ).fetchone()
        if row and row[0] and row[0].startswith("http"):
            return {"app_id": app_id, "header_image": row[0], "source": "db_cache"}

    try:
        # Steam API grąžina JSON su žaidimo duomenimis pagal appid
        # filters=basic,price_overview – imame tik reikalingus laukus
        resp = requests.get(
            STEAM_API_URL,
            params={"appids": app_id, "filters": "basic,price_overview"},
            timeout=5  # 5 sekundžių timeout – Steam API kartais lėta
        )
        data = resp.json().get(str(app_id), {})
        if not data.get("success"):
            return None  # Steam negrąžino duomenų (pvz. žaidimas pašalintas)

        game_data = data["data"]
        header    = game_data.get("header_image", "")

        # price_overview.final saugomas centais (pvz. 1499 = $14.99)
        price_obj = game_data.get("price_overview", {})
        price     = price_obj.get("final", 0) / 100 if price_obj else 0

        # Atnaujiname DB įrašą su naujausiais duomenimis iš Steam
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE games SET header_image = :img, price_final = :price
                WHERE app_id = :id
            """), {"img": header, "price": price, "id": app_id})

        # Mandagus delaymas – vengiame Steam API rate limiting (per daug užklausų)
        time.sleep(0.3)
        return {"app_id": app_id, "header_image": header, "price": price, "source": "steam_api"}

    except Exception as e:
        print(f"⚠️  Steam API klaida (app_id={app_id}): {e}")
        return None


def enrich_batch(app_ids: list[int], max_calls: int = 50) -> dict:
    """
    Partijomis praturtina žaidimų sąrašą iš Steam API.
    max_calls riboja kiek API kvietimų daroma iš karto (apsauga nuo rate limit).
    """
    results = {}
    for i, app_id in enumerate(app_ids[:max_calls]):
        print(f"  [{i+1}/{min(len(app_ids), max_calls)}] app_id={app_id}...")
        res = enrich_with_steam_api(app_id)
        if res:
            results[app_id] = res
    print(f"✅ Praturtinta: {len(results)} žaidimų")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  TF-IDF vs SEMANTIC PALYGINIMAS
# ─────────────────────────────────────────────────────────────────────────────

def compare_tfidf_vs_semantic(queries: list[str], top_k: int = 10) -> pd.DataFrame:
    """
    Palygina TF-IDF ir semantinės paieškos rezultatus toms pačioms užklausoms.

    TF-IDF (Term Frequency–Inverse Document Frequency):
      - Raktiniais žodžiais pagrįsta paieška
      - Greitai, bet nesupranta prasmės (pvz. "shooter" neranda "FPS")

    Semantinė paieška (all-mpnet-base-v2 embeddings):
      - Supranta kontekstą ir sinonimus
      - Lėčiau, bet prasmingesni rezultatai
    """
    print("📂 Krauname žaidimus palyginimui...")
    with engine.connect() as conn:
        # Naudojame LIMIT 10000 – TF-IDF lėtas su visu ~50k žaidimų rinkiniu
        df = pd.read_sql(
            "SELECT app_id, title, about_the_game, tags, genres FROM games LIMIT 10000", conn
        )

    def build_text(row):
        """Ta pati teksto formavimo logika kaip generate_embeddings() – sąžiningas palyginimas."""
        about = str(row["about_the_game"] or "")[:300]
        tags  = str(row["tags"] or "")
        return " ".join(filter(None, [
            str(row["title"] or ""),
            str(row["genres"] or ""),
            tags,
            tags,   # tags du kartus
            about
        ]))

    df["text"] = df.apply(build_text, axis=1)

    # TF-IDF matrica: max_features=20000 – apribojame žodyną, stop_words pašalina
    # anglų kalbos funkcininius žodžius (the, is, at...)
    print("📐 Treniruojame TF-IDF...")
    tfidf        = TfidfVectorizer(max_features=20000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["text"])  # Grąžina sparse matricą

    comparison_rows = []

    for query in queries:
        print(f"\n🔍 Užklausa: '{query}'")

        # TF-IDF paieška: koduojame užklausą ir skaičiuojame cosine panašumą
        q_vec         = tfidf.transform([query])
        scores        = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_tfidf_idx = scores.argsort()[::-1][:top_k]  # Didžiausių balų indeksai
        tfidf_titles  = df.iloc[top_tfidf_idx]["title"].tolist()
        tfidf_scores  = scores[top_tfidf_idx].tolist()

        # Semantinė paieška iš 3.3 funkcijos (su use_cache=False – šviežias rezultatas)
        semantic_results = search(query, top_k=top_k, use_cache=False)
        semantic_titles  = [r["title"] for r in semantic_results]
        semantic_scores  = [r["score"] for r in semantic_results]

        # Overlap – kiek žaidimų abu metodai rado tuose pačiuose top_k
        overlap = len(set(tfidf_titles) & set(semantic_titles))
        print(f"  TF-IDF Top-3:   {tfidf_titles[:3]}")
        print(f"  Semantic Top-3: {semantic_titles[:3]}")
        print(f"  Sutapimas (iš {top_k}): {overlap}")

        # Suformuojame palyginimo eilutes DataFrame'ui
        for rank in range(top_k):
            comparison_rows.append({
                "query":          query,
                "rank":           rank + 1,
                "tfidf_title":    tfidf_titles[rank] if rank < len(tfidf_titles) else "",
                "tfidf_score":    round(tfidf_scores[rank], 4) if rank < len(tfidf_scores) else 0,
                "semantic_title": semantic_titles[rank] if rank < len(semantic_titles) else "",
                "semantic_score": round(semantic_scores[rank], 4) if rank < len(semantic_scores) else 0,
            })

    return pd.DataFrame(comparison_rows)


# ─────────────────────────────────────────────────────────────────────────────
# PALEIDIMAS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("3.1  EMBEDDINGS GENERAVIMAS")
    print("="*60)
    generate_embeddings()

    print("\n" + "="*60)
    print("3.2  FAISS INDEKSAS")
    print("="*60)
    build_faiss_index()

    print("\n" + "="*60)
    print("3.3  PAIEŠKOS FUNKCIJA — TESTAI")
    print("="*60)
    test_queries = [
        "dark fantasy RPG with crafting",
        "relaxing city builder no combat",
        "multiplayer shooter with sci-fi setting",
    ]
    for q in test_queries:
        print(f"\n🔍 '{q}'")
        results = search(q, top_k=5)
        for r in results:
            print(f"  {r['rank']}. {r['title']:<40} score={r['score']:.3f}  ${r['price']}")

    print("\n" + "="*60)
    print("3.4  STEAM API — PRATURTINIMAS")
    print("="*60)
    # Gauname 20 žaidimų iš paieškos ir praturtinome Steam API duomenimis
    sample_ids = [r["app_id"] for r in search("popular action game", top_k=20)]
    enrich_batch(sample_ids, max_calls=20)

    print("\n" + "="*60)
    print("3.5  TF-IDF vs SEMANTIC PALYGINIMAS")
    print("="*60)
    comparison_queries = [
        "dark fantasy RPG with crafting",
        "relaxing city builder",
        "horror survival game",
        "space exploration strategy",
        "indie platformer pixel art",
    ]
    df_comparison = compare_tfidf_vs_semantic(comparison_queries, top_k=10)

    out_path = os.path.join(MODEL_DIR, "tfidf_vs_semantic.csv")
    df_comparison.to_csv(out_path, index=False)
    print(f"\n✅ Palyginimas išsaugotas: {out_path}")
    # Spausdiname tik top 3 kiekvienos užklausos rezultatus peržiūrai
    print(df_comparison[df_comparison["rank"] <= 3].to_string(index=False))