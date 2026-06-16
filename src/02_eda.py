import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Nenaudojame ekrano – grafikai tik išsaugomi į failus
import seaborn as sns
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Įkeliame aplinkos kintamuosius iš .env failo (DB_USER, DB_HOST ir t.t.)
load_dotenv()

# ── Prisijungimas prie duomenų bazės ────────────────────────────────────────
DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)

# ── Keliai ───────────────────────────────────────────────────────────────────
# Bazinis projekto aplankas – ten kur šis failas gyvena
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Grafikai saugomi šalia šio failo, aplanke "eda_output"
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Globali Seaborn tema ir spalvų paletė visiems grafikams
sns.set_theme(style="darkgrid")
COLORS = sns.color_palette("mako", 20)


def get_df(query):
    """Pagalbinė funkcija: vykdo SQL užklausą ir grąžina DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ── 1. Kainų pasiskirstymas ──────────────────────────────────────────────────
def plot_prices():
    df = get_df("SELECT price_final FROM games WHERE price_final > 0 AND price_final < 100")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Žaidimų kainų pasiskirstymas", fontsize=16, fontweight="bold")

    sns.histplot(df["price_final"], bins=50, color=COLORS[2], ax=axes[0])
    axes[0].set_title("Histograma")
    axes[0].set_xlabel("Kaina (USD)")
    axes[0].set_ylabel("Žaidimų skaičius")

    sns.boxplot(x=df["price_final"], color=COLORS[4], ax=axes[1])
    axes[1].set_title("Boxplot")
    axes[1].set_xlabel("Kaina (USD)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_kainos.png"), dpi=150)
    plt.close()
    print("✅ 01_kainos.png išsaugotas")


# ── 2. Žanrų analizė (iš tags stulpelio) ────────────────────────────────────
def plot_genres():
    df = get_df("""
        SELECT tags FROM games
        WHERE tags IS NOT NULL
          AND tags != ''
          AND tags != 'nan'
    """)

    genre_counts = {}
    for row in df["tags"]:
        for g in str(row).split(","):
            g = g.strip()
            if g and g != "nan":
                genre_counts[g] = genre_counts.get(g, 0) + 1

    genre_df = pd.DataFrame(
        sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:20],
        columns=["Žanras", "Skaičius"]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("mako", len(genre_df))
    ax.barh(genre_df["Žanras"][::-1], genre_df["Skaičius"][::-1], color=colors)
    ax.set_title("Top 20 žaidimų žanrų / tagų", fontsize=16, fontweight="bold")
    ax.set_xlabel("Žaidimų skaičius")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_zanrai.png"), dpi=150)
    plt.close()
    print("✅ 02_zanrai.png išsaugotas")


# ── 3. Koreliacijos matrica ──────────────────────────────────────────────────
def plot_correlation():
    df = get_df("""
        SELECT g.price_final, g.positive_ratio, g.user_reviews,
               g.metacritic_score, AVG(r.hours) as avg_playtime
        FROM games g
        JOIN recommendations r ON g.app_id = r.app_id
        WHERE g.price_final < 100
          AND r.hours > 0 AND r.hours < 1000
        GROUP BY g.app_id, g.price_final, g.positive_ratio,
                 g.user_reviews, g.metacritic_score
    """)

    df = df.rename(columns={
        "price_final":     "Kaina",
        "positive_ratio":  "Teigiami %",
        "user_reviews":    "Apžvalgos",
        "metacritic_score":"Metacritic",
        "avg_playtime":    "Avg valandos"
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f",
                cmap="mako", linewidths=0.5, ax=ax)
    ax.set_title("Koreliacijos matrica", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_koreliacija.png"), dpi=150)
    plt.close()
    print("✅ 03_koreliacija.png išsaugotas")


# ── 4. Reitingų pasiskirstymas ───────────────────────────────────────────────
def plot_ratings():
    df = get_df("""
        SELECT rating, COUNT(*) as count
        FROM games
        WHERE rating IS NOT NULL
          AND rating != ''
          AND rating != 'nan'
        GROUP BY rating
        ORDER BY CASE rating
            WHEN 'Overwhelmingly Positive' THEN 1
            WHEN 'Very Positive'           THEN 2
            WHEN 'Positive'                THEN 3
            WHEN 'Mostly Positive'         THEN 4
            WHEN 'Mixed'                   THEN 5
            WHEN 'Mostly Negative'         THEN 6
            WHEN 'Negative'                THEN 7
            WHEN 'Very Negative'           THEN 8
            WHEN 'Overwhelmingly Negative' THEN 9
            ELSE 10
        END
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("mako", len(df))
    ax.barh(df["rating"][::-1], df["count"][::-1], color=colors)
    ax.set_title("Žaidimų reitingų pasiskirstymas", fontsize=16, fontweight="bold")
    ax.set_xlabel("Žaidimų skaičius")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_reitingai.png"), dpi=150)
    plt.close()
    print("✅ 04_reitingai.png išsaugotas")


# ── 5. Vartotojų elgsena – top žaidimai pagal vidutines valandas ─────────────
def plot_user_behavior():
    df = get_df("""
        SELECT g.title, AVG(r.hours) as avg_hours, COUNT(r.user_id) as vartotojai
        FROM recommendations r
        JOIN games g ON r.app_id = g.app_id
        WHERE r.hours > 0 AND r.hours < 1000
          AND g.tags NOT LIKE '%Software%'
          AND g.tags NOT LIKE '%Utilities%'
          AND g.tags NOT LIKE '%Video Production%'
          AND g.tags NOT LIKE '%Animation%'
        GROUP BY g.title
        HAVING COUNT(r.user_id) >= 50
        ORDER BY avg_hours DESC
        LIMIT 30
    """)

    df = df[df["title"].apply(lambda x: str(x).isascii())].head(30)

    fig, ax = plt.subplots(figsize=(12, 12))
    colors = sns.color_palette("mako", len(df))
    ax.barh(df["title"][::-1], df["avg_hours"][::-1], color=colors)
    ax.set_title("Top 30 žaidimų pagal vidutines valandas",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Vidutinės valandos")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_vartotoju_elgsena.png"), dpi=150)
    plt.close()
    print("✅ 05_vartotoju_elgsena.png išsaugotas")


# ── 6. Populiariausi žaidimai pagal vartotojų skaičių ───────────────────────
def plot_popular_games():
    df = get_df("""
        SELECT g.title, COUNT(r.user_id) as vartotojai
        FROM recommendations r
        JOIN games g ON r.app_id = g.app_id
        GROUP BY g.title
        ORDER BY vartotojai DESC
        LIMIT 50
    """)

    df = df[df["title"].apply(lambda x: str(x).isascii())].head(30)

    fig, ax = plt.subplots(figsize=(12, 12))
    colors = sns.color_palette("mako", len(df))
    ax.barh(df["title"][::-1], df["vartotojai"][::-1], color=colors)
    ax.set_title("Top 30 populiariausių žaidimų",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Vartotojų skaičius")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_populiariausi.png"), dpi=150)
    plt.close()
    print("✅ 06_populiariausi.png išsaugotas")


# ── 7. Free-to-play vs mokamų žaidimų santykis ───────────────────────────────
def plot_free_vs_paid():
    df = get_df("""
        SELECT
            CASE WHEN price_final = 0 THEN 'Nemokamas' ELSE 'Mokamas' END as tipas,
            COUNT(*) as count
        FROM games
        GROUP BY tipas
    """)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(df["count"], labels=df["tipas"], autopct="%1.1f%%",
       colors=[COLORS[2], COLORS[6]], startangle=90,
       textprops={"fontsize": 14, "color": "white"})
    ax.set_title("Nemokamų vs mokamų žaidimų santykis",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_free_vs_paid.png"), dpi=150)
    plt.close()
    print("✅ 07_free_vs_paid.png išsaugotas")


# ── 8. Žaidimų išleidimo metai ───────────────────────────────────────────────
def plot_release_years():
    df = get_df("""
        SELECT SUBSTRING(date_release, 1, 4) as metai, COUNT(*) as count
        FROM games
        WHERE date_release IS NOT NULL
          AND date_release != ''
          AND date_release != 'nan'
          AND SUBSTRING(date_release, 1, 4) ~ '^[0-9]{4}$'
          AND SUBSTRING(date_release, 1, 4)::int BETWEEN 2000 AND 2024
        GROUP BY metai
        ORDER BY metai
    """)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("mako", len(df))
    ax.bar(df["metai"], df["count"], color=colors)
    ax.set_title("Žaidimų skaičius pagal išleidimo metus", fontsize=16, fontweight="bold")
    ax.set_xlabel("Metai")
    ax.set_ylabel("Žaidimų skaičius")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_metai.png"), dpi=150)
    plt.close()
    print("✅ 08_metai.png išsaugotas")


# ── Paleidimas ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"📊 Generuojame EDA grafikus...\n")
    print(f"📁 Output: {OUTPUT_DIR}\n")
    plot_prices()        # 1. Kainų histograma + boxplot
    plot_genres()        # 2. Top 20 žanrų/tagų
    plot_correlation()   # 3. Koreliacijos heatmap
    plot_ratings()       # 4. Steam reitingų pasiskirstymas
    plot_user_behavior() # 5. Top žaidimai pagal vidutines valandas
    plot_popular_games() # 6. Top žaidimai pagal vartotojų skaičių
    plot_free_vs_paid()  # 7. Nemokamų vs mokamų žaidimų skritulinė diagrama
    plot_release_years() # 8. Žaidimų skaičius pagal išleidimo metus
    print(f"\n✅ Visi grafikai išsaugoti: {OUTPUT_DIR}")