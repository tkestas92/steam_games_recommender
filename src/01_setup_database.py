# Duomenų apdorojimui lentelėse (DataFrame)
import pandas as pd
# SQLAlchemy: duomenų bazės ryšys ir SQL užklausų vykdymas
from sqlalchemy import create_engine, text
# Psycopg2 funkcija masiniam (batch) duomenų įkėlimui į PostgreSQL (daug greičiau nei standartinis iteravimas)
from psycopg2.extras import execute_values
# Operacinės sistemos funkcijos: aplinkos kintamieji, failų keliai
import os
# Matematiniai skaičiavimai ir trūkstamų (NaN) reikšmių apdorojimas
import numpy as np
# .env failo kintamųjų įkėlimas (saugumo praktika nelaikyti slaptažodžių kode)
from dotenv import load_dotenv

# Užkrauname aplinkos kintamuosius iš .env failo
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PRISIJUNGIMAS
# ─────────────────────────────────────────────────────────────────────────────

# Duomenų bazės URL suformuojamas dinamiškai iš aplinkos kintamųjų
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # Railway / production: pilnas URL su slaptažodžiu
    DB_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
else:
    # Lokalus dev be slaptažodžio
    DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DB_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
else:
    DB_URL = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

# Katalogai, kuriuose laikomi CSV failai ir modeliai
DATA_DIR  = os.getenv("DATA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")

# SQLAlchemy variklis: valdo ryšį su PostgreSQL (echo=False išjungia SQL užklausų loginimą į konsolę)
engine = create_engine(DB_URL, echo=False)

# ─────────────────────────────────────────────────────────────────────────────
# LENTELIŲ SUKŪRIMAS
# ─────────────────────────────────────────────────────────────────────────────

def create_tables():
    """
    Ištrina senas lenteles (jei jos egzistuoja) ir sukuria naujas su atitinkamais 
    duomenų tipais bei ryšiais (Foreign Keys).
    """
    # engine.begin() automatiškai atidaro transakciją ir padaro COMMIT pabaigoje
    with engine.begin() as conn:
        # Ištriname lenteles kaskadiškai/iš eilės, kad išvengtume konfliktų dėl ryšių
        conn.execute(text("DROP TABLE IF EXISTS recommendations;"))
        conn.execute(text("DROP TABLE IF EXISTS users;"))
        conn.execute(text("DROP TABLE IF EXISTS games;"))
        conn.execute(text("DROP TABLE IF EXISTS search_cache;"))
        conn.execute(text("DROP TABLE IF EXISTS model_results;"))
        
        # Pagrindinė žaidimų lentelė
        conn.execute(text("""
            CREATE TABLE games (
                app_id           INTEGER PRIMARY KEY,
                title            TEXT,
                date_release     TEXT,
                win              BOOLEAN,
                mac              BOOLEAN,
                linux            BOOLEAN,
                rating           TEXT,
                positive_ratio   INTEGER,
                user_reviews     INTEGER,
                price_final      FLOAT,
                price_original   FLOAT,
                discount         FLOAT,
                steam_deck       BOOLEAN,
                about_the_game   TEXT,
                genres           TEXT,
                tags             TEXT,
                categories       TEXT,
                developers       TEXT,
                publishers       TEXT,
                header_image     TEXT,
                metacritic_score INTEGER,
                positive         INTEGER,
                negative         INTEGER,
                avg_playtime     INTEGER
            );
        """))
        
        # Vartotojų lentelė
        conn.execute(text("""
            CREATE TABLE users (
                user_id  BIGINT PRIMARY KEY,
                products INTEGER,
                reviews  INTEGER
            );
        """))
        
        # Rekomendacijų (atsiliepimų) lentelė. Susijusi su 'games' ir 'users' lentelėmis.
        conn.execute(text("""
            CREATE TABLE recommendations (
                review_id      INTEGER PRIMARY KEY,
                app_id         INTEGER REFERENCES games(app_id),
                user_id        BIGINT REFERENCES users(user_id),
                helpful        INTEGER,
                funny          INTEGER,
                date           TEXT,
                is_recommended BOOLEAN,
                hours          FLOAT
            );
        """))
        
        # Spartinančioji atmintis (cache) paieškos užklausoms
        conn.execute(text("""
            CREATE TABLE search_cache (
                query_hash  TEXT PRIMARY KEY,
                query_text  TEXT,
                result_json TEXT,
                created_at  TIMESTAMP DEFAULT NOW()
            );
        """))
        
        # Modelių metrikų istorija (Modelių vertinimo palyginimams)
        conn.execute(text("""
            CREATE TABLE model_results (
                id          SERIAL PRIMARY KEY,
                model_name  TEXT,
                precision_k FLOAT,
                recall_k    FLOAT,
                trained_at  TIMESTAMP DEFAULT NOW()
            );
        """))
    print("✅ Lentelės sukurtos")

# ─────────────────────────────────────────────────────────────────────────────
# GAMES ĮKĖLIMAS
# ─────────────────────────────────────────────────────────────────────────────

def load_games():
    """
    Nuskaito pagrindinius ir papildomus žaidimų duomenis iš CSV, 
    apjungia juos, išvalo ir masiškai įkelia į 'games' lentelę.
    """
    print("📂 Skaitome games3.csv...")
    df1 = pd.read_csv(f"{DATA_DIR}/games3.csv")
    df1.columns = df1.columns.str.lower()
    # Užtikriname, kad app_id yra skaičius. 'coerce' paverčia klaidingas reikšmes į NaN
    df1["app_id"] = pd.to_numeric(df1["app_id"], errors="coerce")
    df1 = df1[df1["app_id"].notna()]
    df1["app_id"] = df1["app_id"].astype(int)

    print("📂 Skaitome games_papildomas.csv...")
    # on_bad_lines="skip" ignoruos sugadintas CSV eilutes, kurios trukdytų nuskaitymui
    df2 = pd.read_csv(
        f"{DATA_DIR}/games_papildomas.csv",
        on_bad_lines="skip",
        engine="python",
        encoding="utf-8",
        index_col=0
    )
    df2 = df2.reset_index()
    # Pervadiname stulpelius, kad jie atitiktų mūsų DB schemos pavadinimus
    df2 = df2.rename(columns={
        "index":                    "app_id",
        "About the game":           "about_the_game",
        "Genres":                   "genres",
        "Tags":                     "tags",
        "Categories":               "categories",
        "Developers":               "developers",
        "Publishers":               "publishers",
        "Header image":             "header_image",
        "Metacritic score":         "metacritic_score",
        "Positive":                 "positive",
        "Negative":                 "negative",
        "Average playtime forever": "avg_playtime"
    })

    # Duomenų pavyzdžiai diagnostikai
    print(f"   genres pavyzdys: {df2['genres'].dropna().iloc[0]}")
    print(f"   tags pavyzdys: {df2['tags'].dropna().iloc[0]}")
    print(f"   developers pavyzdys: {df2['developers'].dropna().iloc[0]}")
    print(f"   header_image pavyzdys: {df2['header_image'].dropna().iloc[0][:50]}")

    df2["app_id"] = pd.to_numeric(df2["app_id"], errors="coerce")
    df2 = df2[df2["app_id"].notna()]
    # Saugumas: pašaliname inf reikšmes, jei tokių atsirado po skaičiavimų
    df2 = df2[df2["app_id"] != float("inf")]
    df2["app_id"] = df2["app_id"].astype(int)

    print(f"   df1 app_id pavyzdžiai: {df1['app_id'].head(3).tolist()}")
    print(f"   df2 app_id pavyzdžiai: {df2['app_id'].head(3).tolist()}")

    # Apjungiame DataFrame'us (Left Join pagrindu pagal app_id)
    df = pd.merge(
        df1,
        df2[["app_id", "about_the_game", "genres", "tags", "categories",
             "developers", "publishers", "header_image",
             "metacritic_score", "positive", "negative", "avg_playtime"]],
        on="app_id", how="left"
    )

    print(f"   JOIN po merge — genres ne-null: {df['genres'].notna().sum()}")
    print(f"   JOIN po merge — header_image ne-null: {df['header_image'].notna().sum()}")

    # Užtikriname, kad skaitiniai stulpeliai neturėtų NaN reikšmių (pakeičiame jas 0)
    for col in ["price_final", "price_original", "positive_ratio",
                "user_reviews", "metacritic_score", "positive",
                "negative", "avg_playtime", "discount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Išmetame dublikatus pagal unikalų identifikatorių
    df = df.drop_duplicates(subset="app_id").dropna(subset=["app_id"])
    print(f"   Iš viso žaidimų: {len(df)}")

    # Transformuojame Pandas DataFrame į Python 'tuple' sąrašą.
    # Tai reikalinga norint naudoti greitąjį 'psycopg2.execute_values'
    records = [(
        int(row["app_id"]),
        str(row.get("title") or ""),
        str(row.get("date_release") or ""),
        bool(row.get("win", False)),
        bool(row.get("mac", False)),
        bool(row.get("linux", False)),
        str(row.get("rating") or ""),
        int(row.get("positive_ratio", 0)),
        int(row.get("user_reviews", 0)),
        float(row.get("price_final", 0)),
        float(row.get("price_original", 0)),
        float(row.get("discount", 0)),
        bool(row.get("steam_deck", False)),
        str(row.get("about_the_game") or ""),
        str(row.get("genres") or ""),
        str(row.get("tags") or ""),
        str(row.get("categories") or ""),
        str(row.get("developers") or ""),
        str(row.get("publishers") or ""),
        str(row.get("header_image") or ""),
        int(row.get("metacritic_score", 0)),
        int(row.get("positive", 0)),
        int(row.get("negative", 0)),
        int(row.get("avg_playtime", 0))
    ) for _, row in df.iterrows()]

    # Atidarome tiesioginį ryšį su DB masiniam įkėlimui
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # execute_values sujungia visus records į vieną masinę SQL užklausą
            # ON CONFLICT (app_id) DO NOTHING užtikrina, kad dublikatai nesukels klaidos
            execute_values(cur, """
                INSERT INTO games (
                    app_id, title, date_release, win, mac, linux, rating,
                    positive_ratio, user_reviews, price_final, price_original,
                    discount, steam_deck, about_the_game, genres, tags,
                    categories, developers, publishers, header_image,
                    metacritic_score, positive, negative, avg_playtime
                ) VALUES %s ON CONFLICT (app_id) DO NOTHING;
            """, records, page_size=500)
        raw_conn.commit()
    finally:
        raw_conn.close()
    print("✅ games lentelė užpildyta")
    
# ─────────────────────────────────────────────────────────────────────────────
# USERS ĮKĖLIMAS
# ─────────────────────────────────────────────────────────────────────────────

def load_users(df_rec):
    """
    Filtruoja tik tuos vartotojus, kurie turi bent 50 rekomendacijų (aktyvūs vartotojai),
    ir įkelia juos į DB. Tai sumažina triukšmą rekomendacinėse sistemose.
    """
    print("📂 Filtruojame aktyvius vartotojus (50+ rekomendacijos)...")
    counts = df_rec["user_id"].value_counts()
    
    # Sukuriame aibę (set) su aktyviais vartotojais (skaičiavimo greičiui)
    active_users = set(counts[counts >= 50].index)
    print(f"   Aktyvių vartotojų: {len(active_users)}")

    df = pd.read_csv(f"{DATA_DIR}/users3.csv")
    df.columns = df.columns.str.lower()
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    
    # Paliekame tik tuos vartotojus, kurie patenka į aktyviųjų sąrašą
    df = df[df["user_id"].isin(active_users)].drop_duplicates(subset="user_id")
    df = df.dropna(subset=["user_id"])
    print(f"   Įkeliama: {len(df)}")

    # Paruošiame masiniam įkėlimui
    records = [
        (int(r["user_id"]), int(r.get("products", 0)), int(r.get("reviews", 0)))
        for _, r in df.iterrows()
    ]

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO users (user_id, products, reviews)
                VALUES %s ON CONFLICT (user_id) DO NOTHING;
            """, records, page_size=5000)
        raw_conn.commit()
    finally:
        raw_conn.close()
    print("✅ users lentelė užpildyta")

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS ĮKĖLIMAS
# ─────────────────────────────────────────────────────────────────────────────

def load_recommendations(df_rec):
    """
    Nuskaito atsiliepimus (rekomendacijas) ir išfiltruoja tas eilutes, kurios
    nurodo į neegzistuojančius žaidimus ar vartotojus (Foreign Key apsauga).
    """
    df = df_rec.copy()
    df.columns = df.columns.str.lower()
    df = df.drop_duplicates(subset="review_id")
    df["app_id"]  = pd.to_numeric(df["app_id"],  errors="coerce")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df = df.dropna(subset=["app_id", "user_id", "review_id"])

    # Ištraukiame VISUS jau įkeltų žaidimų ir vartotojų ID iš DB.
    # Tai reikalinga norint išvengti SQL Foreign Key pažeidimų klaidų.
    with engine.connect() as conn:
        valid_apps  = set(r[0] for r in conn.execute(text("SELECT app_id FROM games")).fetchall())
        valid_users = set(r[0] for r in conn.execute(text("SELECT user_id FROM users")).fetchall())

    # Paliekame tik tas rekomendacijas, kurioms DB jau turi žaidimą ir vartotoją
    df = df[df["app_id"].isin(valid_apps) & df["user_id"].isin(valid_users)]
    print(f"   Rekomendacijų (po filtravimo): {len(df)}")

    records = [(
        int(row["review_id"]),
        int(row["app_id"]),
        int(row["user_id"]),
        int(row.get("helpful", 0)),
        int(row.get("funny", 0)),
        str(row.get("date") or ""),
        bool(row.get("is_recommended", False)),
        float(row.get("hours", 0))
    ) for _, row in df.iterrows()]

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO recommendations (
                    review_id, app_id, user_id, helpful, funny,
                    date, is_recommended, hours
                ) VALUES %s ON CONFLICT (review_id) DO NOTHING;
            """, records, page_size=5000)
        raw_conn.commit()
    finally:
        raw_conn.close()
    print("✅ recommendations lentelė užpildyta")

# ─────────────────────────────────────────────────────────────────────────────
# PALEIDIMAS (EXECUTION)
# ─────────────────────────────────────────────────────────────────────────────

# Šis blokas vykdomas tik tada, kai failas paleidžiamas tiesiogiai (ne importuojamas)
if __name__ == "__main__":
    print("🚀 Jungiamės prie PostgreSQL (SQLAlchemy 2.0)...")
    create_tables()
    load_games()
    print("📂 Skaitome recommendations3.csv (vieną kartą)...")
    df_rec = pd.read_csv(f"{DATA_DIR}/recommendations3.csv")
    load_users(df_rec)
    load_recommendations(df_rec)
    print("\n✅ Viskas įkelta į DB!")