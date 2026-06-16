"""
4 etapas — Rekomendacijų sistema
Užduotys:
  4.1  User-Item matrica (scipy sparse)
  4.2  SVD modelis (TruncatedSVD)
  4.3  NCF modelis (PyTorch)
  4.4  Palyginimas: Precision@10, Recall@10 → model_results DB
  4.5  Datasets sujungimas: games3 + recommendations3 per app_id
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── Prisijungimas prie duomenų bazės ─────────────────────────────────────────
DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
MODEL_DIR = os.getenv("MODEL_DIR")  # Aplankas modelių failams (.npy, .pth)
os.makedirs(MODEL_DIR, exist_ok=True)

engine = create_engine(DB_URL, echo=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4.5  DUOMENŲ ĮKĖLIMAS IR SUJUNGIMAS
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Nuskaito recommendations + games iš DB.
    Filtruoja: vartotojai >=50 reviews, žaidimai >=200 reviews.
    Grąžina: df_rec, df_games, id_to_title

    Filtravimo logika:
      - Vartotojai su <50 žaidimų – per mažai duomenų kokybiškam profiliui
      - Žaidimai su <200 apžvalgų – per reti, modelis negali išmokti šablonų
    """
    print("📂 Krauname duomenis iš DB...")

    with engine.connect() as conn:
        df_rec = pd.read_sql("""
            SELECT r.user_id, r.app_id, r.hours, r.is_recommended
            FROM recommendations r
        """, conn)

        df_games = pd.read_sql("""
            SELECT app_id, title, genres, tags, price_final, positive_ratio
            FROM games
        """, conn)

    print(f"   Rekomendacijos (raw): {len(df_rec)}")

    # Filtruojame neaktyvius vartotojus (mažiau nei 50 žaidimų istorijoje)
    user_counts  = df_rec["user_id"].value_counts()
    active_users = user_counts[user_counts >= 50].index
    df_rec       = df_rec[df_rec["user_id"].isin(active_users)]

    # Filtruojame retus žaidimus (mažiau nei 200 apžvalgų duomenų rinkinyje)
    game_counts   = df_rec["app_id"].value_counts()
    popular_games = game_counts[game_counts >= 200].index
    df_rec        = df_rec[df_rec["app_id"].isin(popular_games)]

    print(f"   Po filtravimo: {len(df_rec)}")
    print(f"   Vartotojai: {df_rec['user_id'].nunique()}, Žaidimai: {df_rec['app_id'].nunique()}")

    # 4.5 LEFT JOIN – pridedame žaidimo pavadinimą ir žanrą prie kiekvieno įrašo
    df_merged = df_rec.merge(
        df_games[["app_id", "title", "genres"]],
        on="app_id", how="left"
    )
    print(f"   Sujungtas DataFrame: {df_merged.shape}")

    # Žodynas app_id → title greitam pavadinimų paieškai
    id_to_title = dict(zip(df_games["app_id"], df_games["title"]))
    return df_rec, df_games, id_to_title


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  USER-ITEM MATRICA
# ─────────────────────────────────────────────────────────────────────────────

def build_user_item_matrix(df_rec):
    """
    Sukuria user-item matricą su log(1+hours) kaip svoriu.
    is_recommended=False → rating * 0.1

    Kodėl log transformacija?
      - Vartotojas su 1000h ir 100h turėtų turėti panašų svorį, ne 10x skirtumą
      - log(1+1000) ≈ 6.9, log(1+100) ≈ 4.6 — daug mažesnis skirtumas nei raw
      - Sumažina populiarių žaidimų dominavimą matricoje

    Kodėl 0.1 neigiamoms apžvalgoms?
      - is_recommended=False reiškia kad vartotojas žaidė, bet nerekomenduoja
      - Vis tiek saugome informaciją (žaidė!), bet su mažu svoriu
    """
    print("\n" + "="*60)
    print("4.1  USER-ITEM MATRICA")
    print("="*60)

    df = df_rec.copy()
    # log(1+hours) – clip prie 100 kad ekstremalios reikšmės (10000h) neiškreiptų
    df["rating"] = np.log1p(df["hours"].clip(0, 100))
    # Neigiamos apžvalgos gauna 10x mažesnį svorį
    df.loc[df["is_recommended"] == False, "rating"] *= 0.1

    # Pivot: eilutės = vartotojai, stulpeliai = žaidimai, reikšmės = rating
    # aggfunc="sum" – jei vartotojas turi kelis įrašus tam žaidimui, sumuojame
    user_item = df.pivot_table(
        index="user_id",
        columns="app_id",
        values="rating",
        aggfunc="sum",
        fill_value=0  # Nežaisti = 0, ne NaN
    )

    # Konvertuojame į sparse matricą – taupome atmintį (dauguma reikšmių = 0)
    sparse_matrix = csr_matrix(user_item.values)
    # Tankis: nnz / (vartotojai * žaidimai) – tikėtina ~0.001 (labai reta matrica)
    density = sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])

    print(f"   Matricos dydis: {user_item.shape}")
    print(f"   Tankis: {density:.4f}")
    print(f"   Log transformacija: log(1+hours) naudojama")

    user_ids = list(user_item.index)
    game_ids = list(user_item.columns)

    return sparse_matrix, user_item, user_ids, game_ids


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  SVD MODELIS
# ─────────────────────────────────────────────────────────────────────────────

def train_svd(sparse_matrix, game_ids, id_to_title, n_components=50):
    """
    Treniruoja TruncatedSVD, išsaugo faktorius.
    Grąžina: svd, user_factors, item_factors

    TruncatedSVD (Singular Value Decomposition):
      - Skaldo matricą M ≈ U × Σ × Vᵀ
      - U (user_factors): kiekvienas vartotojas → 50d latentinis vektorius
      - Vᵀ (item_factors): kiekvienas žaidimas → 50d latentinis vektorius
      - n_components=50: latentinių dimensijų skaičius (hiperparametras)
      - Panašūs žaidimai latentinėje erdvėje → panašios rekomendacijos
    """
    print("\n" + "="*60)
    print("4.2  SVD MODELIS")
    print("="*60)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    # fit_transform: išmoksta ir transformuoja vienu žingsniu → user faktoriai
    user_factors = svd.fit_transform(sparse_matrix)
    # components_.T → item faktoriai (žaidimų latentiniai vektoriai)
    item_factors = svd.components_.T

    print(f"   User factors: {user_factors.shape}")   # (n_users, 50)
    print(f"   Item factors: {item_factors.shape}")   # (n_items, 50)
    # Kiek originalo dispersijos paaiškina 50 komponentų (pvz. 0.35 = 35%)
    print(f"   Paaiškinta dispersija: {svd.explained_variance_ratio_.sum():.3f}")

    # Išsaugome faktorius – Flask app juos įkels tiesiai be perkartojamo treniravimo
    np.save(os.path.join(MODEL_DIR, "svd_user_factors.npy"), user_factors)
    np.save(os.path.join(MODEL_DIR, "svd_item_factors.npy"), item_factors)
    np.save(os.path.join(MODEL_DIR, "game_ids.npy"), np.array(game_ids))
    print("   ✅ SVD faktoriai išsaugoti")

    return svd, user_factors, item_factors


def recommend_svd(app_id, item_factors, game_ids, id_to_title, top_k=10, df_games=None):
    """
    SVD rekomendacijos pagal žaidimą su žanro boost.

    Veikimo principas:
      1. Randame tikslinį žaidimą latentinėje erdvėje (item_factors[idx])
      2. Skaičiuojame kosinusinį panašumą su visais kitais žaidimais
      3. Žaidimams su sutampančiais žanrais score padidiname 20% (boost)
      4. Grąžiname top_k panašiausių žaidimų

    Žanro boost pagrindimas:
      - SVD latentinė erdvė ne visada atspindi žanrą tiksliai
      - Explicit boost kompensuoja šį trūkumą
      - 1.2x = 20% – nedidelis, bet pastebimas pagerinimas
    """
    if app_id not in game_ids:
        return []

    idx      = game_ids.index(app_id)
    game_vec = item_factors[idx]  # Tiksliniai žaidimo vektorius

    # Kosinusinis panašumas: dot(A,B) / (||A|| * ||B||)
    # 1e-9 pridedame kad išvengtume dalybos iš nulio
    norms = np.linalg.norm(item_factors, axis=1)
    sims  = item_factors.dot(game_vec) / (norms * np.linalg.norm(game_vec) + 1e-9)
    sims[idx] = -1  # Pašaliname patį žaidimą iš rezultatų

    # Žanro boost – tikriname žanrų persidengimą
    if df_games is not None:
        src = df_games[df_games["app_id"] == app_id]
        if not src.empty:
            # Suformuojame tikslinių žaidimo žanrų aibę
            src_genres = set(str(src.iloc[0]["genres"] or "").lower().split(","))
            src_genres = {g.strip() for g in src_genres if g.strip()}
            for i, gid in enumerate(game_ids):
                if gid == app_id:
                    continue
                row = df_games[df_games["app_id"] == gid]
                if row.empty:
                    continue
                rec_genres = set(str(row.iloc[0]["genres"] or "").lower().split(","))
                rec_genres = {g.strip() for g in rec_genres if g.strip()}
                # Jei bent vienas žanras sutampa → 20% boost
                if src_genres & rec_genres:
                    sims[i] *= 1.2

    # argsort() rikiuoja didėjančia tvarka, [::-1] apverčia → didžiausias pirmas
    top_idx = sims.argsort()[::-1][:top_k]
    return [
        {
            "app_id": game_ids[i],
            "title":  id_to_title.get(game_ids[i], "?"),
            "score":  round(float(sims[i]), 3)
        }
        for i in top_idx
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  NCF MODELIS (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class NCF(nn.Module):
    """
    Neural Collaborative Filtering.
    Vartotojas + žaidimas → embedding → MLP → P(žais)

    Architektūra:
      - user_emb: kiekvienas vartotojas → 64d tankus vektorius (išmokstamas)
      - item_emb: kiekvienas žaidimas  → 64d tankus vektorius (išmokstamas)
      - Concatenate: [user_emb, item_emb] → 128d vektorius
      - MLP: 128 → 128 → 64 → 1 su ReLU aktyvacija
      - Sigmoid: binarinis rezultatas ∈ (0,1) = tikimybė kad vartotojas žais

    Dropout(0.2) – reguliarizacija, 20% neuronų atsitiktinai "išjungiami"
    treniravimo metu, kad modelis nepersitreniruotų (overfitting).
    """
    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)              # (batch, 64)
        i = self.item_emb(item_idx)              # (batch, 64)
        x = torch.cat([u, i], dim=1)             # (batch, 128) – sujungiame
        return self.mlp(x).squeeze()             # (batch,) – binarinis balas


def train_ncf(df_rec, user_ids, game_ids, epochs=20, batch_size=2048, lr=0.001):
    """
    Treniruoja NCF modelį.
    Grąžina: modelis, user_to_idx, item_to_idx

    Negative sampling:
      - Modeliui reikia ir teigiamų (žaidė) ir neigiamų (nežaidė) pavyzdžių
      - is_recommended=True  → label=1 (teigiamas)
      - is_recommended=False → label=0 (neigiamas)
      - Balansuojame: imame tiek pat neigiamų kiek teigiamų (1:1 santykis)

    BCELoss (Binary Cross-Entropy):
      - Standartinė funkcija binarinei klasifikacijai
      - loss = -[y*log(p) + (1-y)*log(1-p)]
      - Adam optimizatorius su lr=0.001 – adaptyvus mokymosi greitis
    """
    print("\n" + "="*60)
    print("4.3  NCF MODELIS (PyTorch)")
    print("="*60)

    # Sukuriame indeksų žodynus – PyTorch Embedding sluoksnis naudoja int indeksus
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {g: i for i, g in enumerate(game_ids)}

    # Filtruojame tik tuos įrašus kurių vartotojai/žaidimai yra mūsų žodynuose
    df = df_rec.copy()
    df = df[df["user_id"].isin(user_to_idx) & df["app_id"].isin(item_to_idx)]
    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["item_idx"] = df["app_id"].map(item_to_idx)
    df["label"]    = (df["is_recommended"] == True).astype(float)  # bool → 0.0/1.0

    # Balansuojame teigiamus ir neigiamus pavyzdžius
    pos       = df[df["label"] == 1]
    neg_count = min(len(pos), len(df[df["label"] == 0]))
    neg       = df[df["label"] == 0].sample(neg_count, random_state=42)
    # Sumaišome teigiamus ir neigiamus (frac=1 = visi, shuffle)
    df_train  = pd.concat([pos, neg]).sample(frac=1, random_state=42)

    print(f"   Train dydis: {len(df_train)} (pos: {len(pos)}, neg: {neg_count})")

    # Konvertuojame į PyTorch tensorius
    user_tensor  = torch.tensor(df_train["user_idx"].values, dtype=torch.long)
    item_tensor  = torch.tensor(df_train["item_idx"].values, dtype=torch.long)
    label_tensor = torch.tensor(df_train["label"].values,    dtype=torch.float)

    n_users = len(user_ids)
    n_items = len(game_ids)
    model     = NCF(n_users, n_items, embed_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binarinė kryžminė entropija

    # DataLoader: automatiškai dalina duomenis į batch'us ir maišo
    dataset = torch.utils.data.TensorDataset(user_tensor, item_tensor, label_tensor)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"   Treniruojame {epochs} epochų...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for u_batch, i_batch, l_batch in loader:
            optimizer.zero_grad()        # Nuvalome gradientus iš praėjusio žingsnio
            pred = model(u_batch, i_batch)
            loss = criterion(pred, l_batch)
            loss.backward()              # Backpropagation – skaičiuojame gradientus
            optimizer.step()             # Atnaujiname svorius pagal gradientus
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0:        # Spausdiname kas 5 epochas
            print(f"   Epocha {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    # Išsaugome tik modelio svorius (state_dict), ne visą modelio objektą
    # Flask app atkurs modelį iš NCF klasės ir įkels šiuos svorius
    model_path = os.path.join(MODEL_DIR, "ncf_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"   ✅ NCF modelis išsaugotas: {model_path}")

    return model, user_to_idx, item_to_idx


def recommend_ncf(app_id, model, item_to_idx, game_ids, id_to_title, top_k=10):
    """
    NCF rekomendacijos pagal item embedding kosinusinį panašumą.

    Naudoja išmoktas item embeddingų reikšmes iš NCF modelio.
    Skirtingai nei SVD, čia naudojame neurono išmoktas reprezentacijas –
    jose užkoduota kolaboratyvinė informacija (kas su kuo žaidė).

    model.eval() + torch.no_grad():
      - eval() išjungia Dropout (visi neuronai aktyvūs per inferenciją)
      - no_grad() neskaičiuoja gradientų – greičiau ir taupiau atmintį
    """
    if app_id not in item_to_idx:
        return []

    model.eval()
    with torch.no_grad():
        # Gauname visų žaidimų embeddings iš išmoktų modelio svorių
        all_item_idx  = torch.arange(len(game_ids), dtype=torch.long)
        all_embeddings = model.item_emb(all_item_idx).numpy()  # (n_items, 64)

        target_idx = item_to_idx[app_id]
        target_emb = all_embeddings[target_idx]  # Tiksliniai žaidimo embedding

        # Kosinusinis panašumas tarp tikslinį ir visų kitų žaidimų
        norms       = np.linalg.norm(all_embeddings, axis=1)
        target_norm = np.linalg.norm(target_emb)
        sims        = all_embeddings.dot(target_emb) / (norms * target_norm + 1e-9)
        sims[target_idx] = -1  # Pašaliname patį žaidimą

        top_idx = sims.argsort()[::-1][:top_k]

    return [
        {
            "app_id": game_ids[i],
            "title":  id_to_title.get(game_ids[i], "?"),
            "score":  float(sims[i])
        }
        for i in top_idx
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4.4  PALYGINIMAS: Precision@10, Recall@10
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_models(svd_recs_fn, ncf_recs_fn, user_item, game_ids, test_games, k=10):
    """
    Precision@K ir Recall@K abiem modeliams.

    "Relevant" žaidimo apibrėžimas:
      Jei du žaidimai turi >=10 bendrų vartotojų user_item matricoje,
      laikome juos "relevantiškais" vienas kitam. Tai proxy metrika –
      tikrai relevancei reiktų eksplicitinių vartotojo įvertinimų.

    Precision@K = (rastų relevantiškų) / K
      → Kiek iš K rekomendacijų buvo tikrai geros?

    Recall@K = (rastų relevantiškų) / (visų relevantiškų)
      → Kiek visų galimų gerų žaidimų modelis rado?
    """
    print("\n" + "="*60)
    print("4.4  PALYGINIMAS: Precision@10, Recall@10")
    print("="*60)

    results = {
        "svd": {"precision": [], "recall": []},
        "ncf": {"precision": [], "recall": []}
    }

    for app_id in test_games:
        if app_id not in game_ids:
            continue
        idx          = game_ids.index(app_id)
        # Visi vartotojai kurie žaidė šį žaidimą (rating > 0 matricoje)
        users_played = set(user_item.index[user_item.iloc[:, idx] > 0])
        if len(users_played) < 10:
            continue  # Per mažai duomenų patikimam vertinimui

        for model_name, recs_fn in [("svd", svd_recs_fn), ("ncf", ncf_recs_fn)]:
            recs     = recs_fn(app_id)[:k]
            relevant = 0
            for r in recs:
                if r["app_id"] not in game_ids:
                    continue
                r_idx      = game_ids.index(r["app_id"])
                users_also = set(user_item.index[user_item.iloc[:, r_idx] > 0])
                # Relevantiška jei >=10 tų pačių vartotojų žaidė abu žaidimus
                if len(users_played & users_also) >= 10:
                    relevant += 1
            p       = relevant / k
            r_score = relevant / max(len(users_played), 1)
            results[model_name]["precision"].append(p)
            results[model_name]["recall"].append(r_score)

    # Apskaičiuojame vidurkius per visus test žaidimus
    metrics = {}
    for model_name in ["svd", "ncf"]:
        p = np.mean(results[model_name]["precision"]) if results[model_name]["precision"] else 0
        r = np.mean(results[model_name]["recall"])    if results[model_name]["recall"]    else 0
        metrics[model_name] = {"precision_k": round(p, 4), "recall_k": round(r, 4)}
        print(f"   {model_name.upper():4s} — Precision@{k}: {p:.4f}  Recall@{k}: {r:.4f}")

    return metrics


def save_metrics_to_db(metrics):
    """Išsaugo modelių metrikas į model_results lentelę palyginimui UI."""
    with engine.begin() as conn:
        for model_name, m in metrics.items():
            conn.execute(text("""
                INSERT INTO model_results (model_name, precision_k, recall_k)
                VALUES (:name, :p, :r)
            """), {"name": model_name, "p": m["precision_k"], "r": m["recall_k"]})
    print("   ✅ Metrikos išsaugotos į DB")


# ─────────────────────────────────────────────────────────────────────────────
# PALEIDIMAS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Įkeliame ir filtruojame duomenis iš DB
    df_rec, df_games, id_to_title = load_data()

    # 4.1 Sukuriame sparse user-item matricą su log svoriais
    sparse_matrix, user_item, user_ids, game_ids = build_user_item_matrix(df_rec)

    # 4.2 Treniruojame SVD ir išsaugome faktorius
    svd, user_factors, item_factors = train_svd(sparse_matrix, game_ids, id_to_title)

    # 4.3 Treniruojame NCF neuroninį tinklą
    model, user_to_idx, item_to_idx = train_ncf(df_rec, user_ids, game_ids, epochs=20)

    # Sukuriame lambda funkcijas su užfiksuotais parametrais – patogiau perduoti
    svd_recs = lambda app_id: recommend_svd(app_id, item_factors, game_ids, id_to_title, df_games=df_games)
    ncf_recs = lambda app_id: recommend_ncf(app_id, model, item_to_idx, game_ids, id_to_title)

    # Rankiniai testai su žinomais žaidimais
    test_games_named = {
        "Dota 2":              570,
        "Counter-Strike: GO":  730,
        "Skyrim":              489830
    }
    print("\n" + "="*60)
    print("REKOMENDACIJŲ TESTAI")
    print("="*60)
    for name, app_id in test_games_named.items():
        print(f"\n🎮 '{name}'")
        print("  SVD:")
        for r in svd_recs(app_id)[:5]:
            print(f"    {r['title']:<40} {r['score']:.3f}")
        print("  NCF:")
        for r in ncf_recs(app_id)[:5]:
            print(f"    {r['title']:<40} {r['score']:.3f}")

    # 4.4 Kiekybinis modelių įvertinimas
    test_app_ids = list(test_games_named.values())
    metrics = evaluate_models(svd_recs, ncf_recs, user_item, game_ids, test_app_ids, k=10)

    # Išsaugome metrikas į DB (Flask compare.html jas nuskaito)
    save_metrics_to_db(metrics)

    print("\n✅ 4 etapas baigtas!")