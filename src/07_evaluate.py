"""
07_evaluate.py — sąžiningas modelių vertinimas
================================================

Kam tai skirta
--------------
Senasis /compare puslapis lygino SVD ir NCF vienu skaičiumi, nors tai
skirtingi uždaviniai:

  • NCF   — klasifikatorius: "ar ŠIS vartotojas rekomenduos ŠĮ žaidimą?"
  • SVD   — panašumo modelis: "kokie žaidimai panašūs į tuos, kuriuos žmogus mėgsta?"

Todėl čia kiekvienas modelis vertinamas dviem būdais, visada ant TOS PAČIOS
test aibės ir su paprastais baseline'ais (be jų skaičius neturi atskaitos taško):

  A) Klasifikacija (patinka / nepatinka)   → Accuracy, F1, AUC
       palyginama su: "visada rekomenduoja", "žaidimo vidutinis įvertinimas"
  B) Reitingavimas (Top-10 rekomendacijos) → Recall@10, NDCG@10, HitRate@10
       palyginama su: "populiariausi žaidimai"

Test aibė yra TOKIA PATI kaip NCF v28 treniravimo metu (kodas nukopijuotas iš
ncf_v28_final.ipynb: tie patys filtrai, tas pats random_state=42), todėl
išsaugotas NCF modelis nemato test duomenų.
SVD čia perkeliamas treniruojamas tik ant train dalies (kitaip jis "matytų"
test duomenis).

Paleidimas
----------
    python src/07_evaluate.py --data-dir /kelias/iki/csv --model-dir ./models

  --data-dir  aplankas su recommendations3.csv, games3.csv, games_papildomas.csv
              (tie patys failai kaip NCF notebook'e)
  --model-dir aplankas su ncf_v28_final.pth ir ncf_v28_final_metadata.pkl
  --skip-ncf  praleisti NCF (jei nėra PyTorch / modelio failų)

Išvestis: app/static/shap/model_eval.json — jį rodo /compare puslapis.
"""

import argparse
import json
import os
# macOS (Apple Silicon): torch + numpy/Accelerate kartu gali sukelti "segmentation fault".
# Šie kintamieji turi būti nustatyti PRIEŠ importuojant numpy/torch, todėl jie čia, o ne terminale.
import sys
if sys.platform == "darwin":
    for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import time
from datetime import date

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

# ── Hiperparametrai: turi sutapti su NCF v28 notebook'u ─────────────────────
USER_MIN = 50
GAME_MIN = 1000
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ── Reitingavimo vertinimo nustatymai ───────────────────────────────────────
K = 10               # Top-K rekomendacijų
MIN_TRAIN_POS = 5    # vertiname tik vartotojus, turinčius bent tiek "patikusių" train'e
MAX_EVAL_USERS = 3000  # greičiui: atsitiktinė vartotojų imtis


# ─────────────────────────────────────────────────────────────────────────────
# 1. DUOMENYS — identiška NCF v28 notebook'ui
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir):
    print("📂 Krauname duomenis iš CSV...")
    # usecols taupo atmintį (pilnas CSV turi ~41 mln. eilučių); eilučių tvarka nesikeičia
    df_rec = pd.read_csv(os.path.join(data_dir, "recommendations3.csv"),
                         usecols=["user_id", "app_id", "hours", "is_recommended"])
    df_games = pd.read_csv(os.path.join(data_dir, "games3.csv"))

    df_rec = df_rec[["user_id", "app_id", "hours", "is_recommended"]].copy()
    print(f"   Raw: {len(df_rec)} eilučių")

    user_counts = df_rec["user_id"].value_counts()
    df_rec = df_rec[df_rec["user_id"].isin(user_counts[user_counts >= USER_MIN].index)]
    game_counts = df_rec["app_id"].value_counts()
    df_rec = df_rec[df_rec["app_id"].isin(game_counts[game_counts >= GAME_MIN].index)]
    print(f"   Po filtravimo: {len(df_rec)} eilučių  "
          f"(vartotojai {df_rec['user_id'].nunique()}, žaidimai {df_rec['app_id'].nunique()})")

    df_rec["hours_log"] = np.log1p(df_rec["hours"].clip(0, 1000))

    df_games = df_games[["app_id", "price_final", "positive_ratio"]].copy()
    df_games["price_final"] = df_games["price_final"].fillna(0)
    df_games["positive_ratio"] = df_games["positive_ratio"].fillna(df_games["positive_ratio"].median())
    price_max = df_games["price_final"].max()
    df_games["price_norm"] = df_games["price_final"] / (price_max + 1e-9)
    df_games["ratio_norm"] = df_games["positive_ratio"] / 100.0

    df_pap = pd.read_csv(os.path.join(data_dir, "games_papildomas.csv"))
    df_pap = df_pap[["AppID", "Genres"]].copy()
    df_pap = df_pap.rename(columns={"AppID": "app_id", "Genres": "genres"})
    df_pap["app_id"] = pd.to_numeric(df_pap["app_id"], errors="coerce")
    df_pap = df_pap[df_pap["app_id"].notna() & np.isfinite(df_pap["app_id"])]
    df_pap["app_id"] = df_pap["app_id"].astype(int)
    df_pap["genres"] = df_pap["genres"].fillna("")
    df_pap["genres_list"] = df_pap["genres"].apply(
        lambda x: [g.strip().lower() for g in str(x).split(",") if g.strip()]
    )
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df_pap["genres_list"])
    genre_cols = [f"genre_{g}" for g in mlb.classes_]
    df_genres = pd.DataFrame(genres_encoded, columns=genre_cols, index=df_pap.index)
    df_pap = pd.concat([df_pap[["app_id"]], df_genres], axis=1)

    df_merged = df_rec.merge(df_games[["app_id", "price_norm", "ratio_norm"]], on="app_id", how="left")
    df_merged = df_merged.merge(df_pap, on="app_id", how="left")
    for col in ["price_norm", "ratio_norm"] + genre_cols:
        df_merged[col] = df_merged[col].fillna(0)
    return df_merged, genre_cols


def split_data(df):
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    split_idx = int(len(df) * (1 - TEST_RATIO))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. METRIKOS
# ─────────────────────────────────────────────────────────────────────────────
def classification_metrics(scores, labels, threshold=0.5):
    """Accuracy, F1 (teigiamai klasei) ir AUC iš tikimybių."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(float)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    # AUC neapibrėžtas jei score konstanta → 0.5 (atsitiktinis spėjimas)
    auc = 0.5 if np.ptp(scores) == 0 else float(roc_auc_score(labels, scores))
    return {
        "accuracy": round(float((pred == labels).mean()), 4),
        "f1": round(float(f1), 4),
        "auc": round(auc, 4),
    }


def ranking_metrics(score_matrix, seen_mask, relevant_lists, k=K):
    """
    score_matrix:   (n_users, n_items) — kuo didesnis, tuo labiau rekomenduojama
    seen_mask:      (n_users, n_items) bool — jau matyti train'e (neįtraukiami)
    relevant_lists: kiekvienam vartotojui — test'e patikusių žaidimų indeksai
    """
    sm = np.where(seen_mask, -np.inf, score_matrix)
    recalls, ndcgs, hits = [], [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for row, rel in zip(sm, relevant_lists):
        top = np.argpartition(-row, k)[:k]
        top = top[np.argsort(-row[top])]
        rel_set = set(rel)
        hit_flags = np.array([1.0 if t in rel_set else 0.0 for t in top])
        n_hit = hit_flags.sum()
        recalls.append(n_hit / len(rel_set))
        hits.append(1.0 if n_hit > 0 else 0.0)
        dcg = (hit_flags * discounts).sum()
        idcg = discounts[: min(len(rel_set), k)].sum()
        ndcgs.append(dcg / idcg)
    return {
        "recall": round(float(np.mean(recalls)), 4),
        "ndcg": round(float(np.mean(ndcgs)), 4),
        "hit": round(float(np.mean(hits)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELIAI
# ─────────────────────────────────────────────────────────────────────────────
def build_svd_item_factors(train_df, user_to_idx, item_to_idx, n_components=50):
    """
    SVD pagal tą pačią receptą kaip 04_recommend.py:
      rating = log(1+hours), neigiamos apžvalgos ×0.1; TruncatedSVD(50).
    Treniruojama TIK ant train dalies.
    """
    rating = np.log1p(train_df["hours"].clip(0, 100).values)
    rating = np.where(train_df["is_recommended"].values == True, rating, rating * 0.1)  # noqa: E712
    rows = train_df["user_id"].map(user_to_idx).values
    cols = train_df["app_id"].map(item_to_idx).values
    mat = csr_matrix((rating, (rows, cols)), shape=(len(user_to_idx), len(item_to_idx)))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    svd.fit(mat)
    factors = svd.components_.T  # (n_items, n_components)
    norms = np.linalg.norm(factors, axis=1, keepdims=True) + 1e-9
    return factors / norms


def ncf_scores(model_dir, n_items, item_feat, n_users):
    """NCF tikimybės visiems (vartotojas × žaidimas) deriniams. Reikia PyTorch."""
    import torch
    import torch.nn as nn

    with open(os.path.join(model_dir, "ncf_v28_final_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    class NCF(nn.Module):  # ta pati architektūra kaip notebook'e
        def __init__(self, n_users, n_items, embed_dim, n_features):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, embed_dim)
            self.item_emb = nn.Embedding(n_items, embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2 + n_features, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
            )

        def forward(self, u, i, f):
            return self.mlp(torch.cat([self.user_emb(u), self.item_emb(i), f], dim=1)).squeeze(dim=-1)

    if meta["n_users"] != n_users or meta["n_items"] != n_items:
        raise SystemExit(
            f"❌ Duomenys nesutampa su išsaugotu modeliu: modelis turi "
            f"{meta['n_users']} vartotojų / {meta['n_items']} žaidimų, o CSV po filtravimo — "
            f"{n_users} / {n_items}. Naudok TUOS PAČIUS CSV, kuriais mokytas v28."
        )

    model = NCF(meta["n_users"], meta["n_items"], meta["embed_dim"], meta["n_features"])
    model.load_state_dict(torch.load(os.path.join(model_dir, "ncf_v28_final.pth"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    feat_t = torch.tensor(item_feat, dtype=torch.float)

    def predict(u_idx, i_idx):
        """Prognozuoja poromis (u_idx[j], i_idx[j]) paketais."""
        out = []
        with torch.no_grad():
            for s in range(0, len(u_idx), 65536):
                u = torch.tensor(u_idx[s:s + 65536], dtype=torch.long)
                i = torch.tensor(i_idx[s:s + 65536], dtype=torch.long)
                out.append(model(u, i, feat_t[i]).numpy())
        return np.concatenate(out)

    return predict


# ─────────────────────────────────────────────────────────────────────────────
# 4. PAGRINDINĖ PROGRAMA
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--out", default=os.path.join("app", "static", "shap", "model_eval.json"))
    ap.add_argument("--skip-ncf", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    df, genre_cols = load_data(args.data_dir)
    feature_cols = ["price_norm", "ratio_norm"] + genre_cols
    train_df, test_df = split_data(df)

    all_users = sorted(df["user_id"].unique())
    all_items = sorted(df["app_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {g: i for i, g in enumerate(all_items)}
    n_users, n_items = len(all_users), len(all_items)

    item_feat = (df.drop_duplicates("app_id").set_index("app_id")[feature_cols]
                 .loc[all_items].values.astype(np.float32))

    predict = None
    if not args.skip_ncf:
        predict = ncf_scores(args.model_dir, n_items, item_feat, n_users)

    # ── A) KLASIFIKACIJA (natūralus test) ───────────────────────────────────
    print("\n=== A) Klasifikacija: ar vartotojui patiks žaidimas? ===")
    t_u = test_df["user_id"].map(user_to_idx).values
    t_i = test_df["app_id"].map(item_to_idx).values
    t_y = (test_df["is_recommended"] == True).astype(float).values  # noqa: E712

    train_y = (train_df["is_recommended"] == True).astype(float)  # noqa: E712
    game_rate = train_y.groupby(train_df["app_id"].map(item_to_idx)).mean()
    global_rate = float(train_y.mean())
    game_rate_arr = np.full(n_items, global_rate)
    game_rate_arr[game_rate.index.values] = game_rate.values

    classification = [
        {"model": "Visada „rekomenduoja“", **classification_metrics(np.ones(len(t_y)), t_y)},
        {"model": "Žaidimo vidutinis įvertinimas", **classification_metrics(game_rate_arr[t_i], t_y)},
    ]
    if predict is not None:
        classification.append({"model": "NCF v28", **classification_metrics(predict(t_u, t_i), t_y)})
    for r in classification:
        print(f"   {r['model']:<32} acc={r['accuracy']}  F1={r['f1']}  AUC={r['auc']}")

    # ── B) REITINGAVIMAS (Top-K) ────────────────────────────────────────────
    print(f"\n=== B) Reitingavimas: Top-{K} rekomendacijos ===")
    tr_u = train_df["user_id"].map(user_to_idx).values
    tr_i = train_df["app_id"].map(item_to_idx).values
    tr_pos = train_df["is_recommended"].values == True  # noqa: E712

    seen = np.zeros((n_users, n_items), dtype=bool)
    seen[tr_u, tr_i] = True                              # viskas, ką vartotojas matė train'e
    pos_train = csr_matrix((np.ones(tr_pos.sum()), (tr_u[tr_pos], tr_i[tr_pos])), shape=(n_users, n_items))
    pos_train.data[:] = 1.0                              # dublikatai → 1

    te_pos = t_y == 1
    test_pos_by_user = {}
    for u, i in zip(t_u[te_pos], t_i[te_pos]):
        if not seen[u, i]:                               # nauja vartotojui → verta rekomenduoti
            test_pos_by_user.setdefault(u, []).append(i)

    train_pos_count = np.asarray(pos_train.sum(axis=1)).ravel()
    eligible = [u for u in test_pos_by_user if train_pos_count[u] >= MIN_TRAIN_POS]
    rng = np.random.default_rng(RANDOM_SEED)
    if len(eligible) > MAX_EVAL_USERS:
        eligible = list(rng.choice(eligible, MAX_EVAL_USERS, replace=False))
    eligible = np.array(sorted(eligible))
    relevant = [test_pos_by_user[u] for u in eligible]
    print(f"   Vertinama vartotojų: {len(eligible)}")

    seen_e = seen[eligible]

    # populiarumas: kiek kartų žaidimas patiko train'e
    pop = np.asarray(pos_train.sum(axis=0)).ravel()
    pop_scores = np.tile(pop, (len(eligible), 1))

    # SVD: vartotojo profilis = patikusių žaidimų vektorių vidurkis
    item_vec = build_svd_item_factors(train_df, user_to_idx, item_to_idx)
    profile = pos_train[eligible] @ item_vec
    profile = profile / (np.linalg.norm(profile, axis=1, keepdims=True) + 1e-9)
    svd_scores = profile @ item_vec.T

    ranking = [
        {"model": "Populiariausi žaidimai", **ranking_metrics(pop_scores, seen_e, relevant)},
        {"model": "SVD (panašūs žaidimai)", **ranking_metrics(svd_scores, seen_e, relevant)},
    ]
    if predict is not None:
        u_rep = np.repeat(eligible, n_items)
        i_rep = np.tile(np.arange(n_items), len(eligible))
        ncf_mat = predict(u_rep, i_rep).reshape(len(eligible), n_items)
        ranking.append({"model": "NCF v28", **ranking_metrics(ncf_mat, seen_e, relevant)})
    for r in ranking:
        print(f"   {r['model']:<28} Recall@{K}={r['recall']}  NDCG@{K}={r['ndcg']}  Hit@{K}={r['hit']}")

    # ── Išsaugome ───────────────────────────────────────────────────────────
    result = {
        "generated": date.today().isoformat(),
        "dataset": {
            "users": n_users,
            "games": n_items,
            "interactions": int(len(df)),
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "positive_rate": round(float(t_y.mean()), 4),
            "user_min_reviews": USER_MIN,
            "game_min_reviews": GAME_MIN,
        },
        "classification": classification,
        "ranking": {"k": K, "eval_users": int(len(eligible)), "models": ranking},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Išsaugota: {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()