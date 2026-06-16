"""
5 etapas — RAG + Gemini
Užduotys:
  5.1  Gemini API setup + testo užklausa
  5.2  RAG pipeline — FAISS top-5 → kontekstas → Gemini → paaiškinimas
  5.3  Prompt engineering — optimizuotas prompt šablonas
"""

import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

# ── Prisijungimas prie duomenų bazės ─────────────────────────────────────────
DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
MODEL_DIR  = os.getenv("MODEL_DIR")   # Aplankas FAISS indeksui ir app_ids.npy
GEMINI_KEY = os.getenv("GEMINI_API_KEY")  # Google AI Studio API raktas

engine = create_engine(DB_URL, echo=False)


# ─────────────────────────────────────────────────────────────────────────────
# 5.1  GEMINI API SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_gemini():
    """
    Inicializuoja Gemini klientą ir atlieka testo užklausą.

    google-genai biblioteka naudoja naują Client() API (ne senąjį configure()).
    gemini-2.5-flash – greitas ir pigus modelis, tinka RAG paaiškinimams.
    """
    print("\n" + "="*60)
    print("5.1  GEMINI API SETUP")
    print("="*60)

    # Sukuriame klientą su API raktu iš .env
    client = genai.Client(api_key=GEMINI_KEY)

    # Paprastas testo kvietimas – patikriname ar API raktas veikia
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'Gemini API works!' in one sentence."
    )
    print(f"   ✅ Gemini atsakė: {response.text.strip()}")
    return client


# ─────────────────────────────────────────────────────────────────────────────
# 5.2  RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# Globalūs objektai – lazy loading, įkeliami tik pirmą kartą kai reikia
_st_model    = None
_faiss_index = None
_app_ids     = None

def _load_assets():
    """Įkelia sentence-transformer modelį ir FAISS indeksą į atmintį."""
    global _st_model, _faiss_index, _app_ids
    if _st_model is None:
        print("   🤖 Krauname sentence-transformers modelį...")
        # all-MiniLM-L6-v2 – lengvesnis modelis nei mpnet, tinka greitesniam RAG kontekstui
        _st_model = SentenceTransformer("all-mpnet-base-v2")
    if _faiss_index is None:
        _faiss_index = faiss.read_index(os.path.join(MODEL_DIR, "games.index"))
        _app_ids     = np.load(os.path.join(MODEL_DIR, "app_ids.npy"))


def get_context(query: str, top_k: int = 5) -> list[dict]:
    """
    RAG konteksto gavimas: FAISS paieška → žaidimų info iš DB.
    Grąžina top_k žaidimų su title, genres, tags, about, positive_ratio.

    Reranking logika:
      - FAISS grąžina top_k*3 kandidatų (su rezervu)
      - Paskui rerankiname pagal: semantic 70% + popularity 30%
      - Taip Gemini gauna ir semantiškai artimus, ir gerai įvertintus žaidimus
    """
    _load_assets()

    # Koduojame užklausą į vektorių ir normalizuojame (kaip indekso vektoriai)
    vec = _st_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    distances, indices = _faiss_index.search(vec, top_k * 3)
    found_ids = [int(_app_ids[i]) for i in indices[0] if i != -1]

    # Gauname detalią žaidimų informaciją iš DB – tai bus Gemini kontekstas
    with engine.connect() as conn:
        placeholders = ",".join(str(i) for i in found_ids)
        rows = conn.execute(text(f"""
            SELECT app_id, title, genres, tags, about_the_game,
                   positive_ratio, price_final, rating
            FROM games
            WHERE app_id IN ({placeholders})
        """)).fetchall()

    # Rerankiname: semantinis panašumas + populiarumas
    id_to_row = {r[0]: r for r in rows}
    results   = []
    for app_id, dist in zip(found_ids, distances[0]):
        if app_id not in id_to_row:
            continue
        r = id_to_row[app_id]
        # dist ∈ [0,2] → semantic_score ∈ [0,1]
        semantic_score   = float(1 - dist / 2)
        # positive_ratio DB saugomas kaip 0–100, normalizuojame į 0–1
        popularity_score = float(r[5] or 0) / 100.0
        # 70/30 svoris: semantika svarbesnė, bet populiarumas padeda filtruoti šiukšles
        final_score = semantic_score * 0.7 + popularity_score * 0.3
        results.append({
            "app_id":         r[0],
            "title":          r[1],
            "genres":         r[2] or "",
            "tags":           (r[3] or "")[:200],   # Trumpiname – Gemini kontekstas ribotas
            "about":          (r[4] or "")[:300],   # Aprašymą trumpiname iki 300 simbolių
            "positive_ratio": r[5],
            "price":          r[6],
            "rating":         r[7],
            "score":          final_score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]  # Grąžiname tik top_k po reranking


def build_prompt(query: str, context: list[dict], language: str = "LT") -> str:
    """
    5.3 Optimizuotas prompt šablonas.

    Prompt engineering principai čia:
      - Aiški rolė: "Tu esi žaidimų ekspertas" – nurodo modelio perspektyvą
      - Struktūrizuotas kontekstas: sunumeruoti žaidimai su faktais
      - Konkreti užduotis: 3 aiškiai suformuluoti punktai
      - Ilgio apribojimas: "max 150 žodžių" – vengiame perteklinių atsakymų
      - Kalbos pasirinkimas: LT arba EN pagal poreikį

    Kodėl about[:200] trumpinamas dar kartą nors get_context jau trumpino?
      - get_context grąžina [:300], čia papildomai sutrumpinamas prompt dydžiui
    """
    games_text = ""
    for i, g in enumerate(context, 1):
        games_text += (
            f"{i}. {g['title']} "
            f"(žanrai: {g['genres']}, "
            f"įvertinimas: {g['positive_ratio']}%, "
            f"kaina: ${g['price']:.2f})\n"
            f"   Aprašymas: {g['about'][:200]}\n\n"
        )

    if language == "LT":
        prompt = f"""Tu esi žaidimų ekspertas. Vartotojas ieško: "{query}"

Rasti žaidimai:
{games_text}
Užduotis:
1. Paaiškink kodėl šie žaidimai tinka paieškai (2-3 sakiniai)
2. Išskirk geriausią pasirinkimą ir kodėl
3. Paminėk ką vartotojas gali tikėtis žaisdamas

Atsakyk lietuviškai, glaustai (max 150 žodžių)."""
    else:
        prompt = f"""You are a game expert. User searches for: "{query}"

Found games:
{games_text}
Task:
1. Explain why these games match the search (2-3 sentences)
2. Highlight the best choice and why
3. Mention what the user can expect while playing

Answer concisely in English (max 150 words)."""

    return prompt


def rag_explain(query: str, client, language: str = "LT", top_k: int = 5) -> dict:
    """
    Pilnas RAG (Retrieval-Augmented Generation) pipeline:

    RAG = Retrieve + Augment + Generate
      1. Retrieve: FAISS paieška → top_k relevantiškų žaidimų (kontekstas)
      2. Augment:  Įterpiame kontekstą į prompt šabloną
      3. Generate: Gemini generuoja paaiškinimą remdamasis kontekstu

    Kodėl RAG, o ne tiesiog Gemini be konteksto?
      - Gemini nežino Steam žaidimų duomenų bazės
      - Su RAG pateikiame faktus (kainas, įvertinimus, aprašymus) – modelis jais remiasi
      - Mažesnis hallucination tikimybė: modelis kalba apie konkrečius rastus žaidimus

    Grąžina: {"query", "games", "explanation", "prompt"}
    """
    # 1. Retrieve – gauname semantiškai artimus žaidimus
    context = get_context(query, top_k=top_k)

    # 2. Augment – suformuojame prompt su kontekstu
    prompt  = build_prompt(query, context, language=language)

    # 3. Generate – Gemini generuoja paaiškinimą
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "query":       query,
        "games":       context,
        "explanation": response.text.strip(),
        "prompt":      prompt,  # Saugome prompt – naudinga debug/logavimui
    }


# ─────────────────────────────────────────────────────────────────────────────
# PALEIDIMAS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 5.1 Inicializuojame Gemini klientą
    client = setup_gemini()

    # 5.2 + 5.3 RAG pipeline testai su skirtingomis užklausomis ir kalbomis
    print("\n" + "="*60)
    print("5.2 + 5.3  RAG PIPELINE TESTAI")
    print("="*60)

    test_queries = [
        ("dark fantasy RPG with crafting", "LT"),  # Lietuviškas paaiškinimas
        ("relaxing city builder no combat", "LT"),
        ("multiplayer shooter sci-fi",      "EN"),  # Angliškas paaiškinimas
    ]

    for query, lang in test_queries:
        print(f"\n🔍 Užklausa: '{query}' [{lang}]")
        print("-" * 50)

        result = rag_explain(query, client, language=lang, top_k=5)

        # Spausdiname rastus žaidimus (kontekstą)
        print("📋 Rasti žaidimai:")
        for g in result["games"]:
            print(f"   • {g['title']} ({g['positive_ratio']}% teigiamų, ${g['price']:.2f})")

        # Spausdiname Gemini sugeneruotą paaiškinimą
        print(f"\n💬 Gemini paaiškinimas:\n{result['explanation']}")
        print()

    print("✅ 5 etapas baigtas!")