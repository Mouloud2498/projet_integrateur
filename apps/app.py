"""
app.py — BooksMatch
API Flask — système de recommandation de livres hybride SVD + Jina + Popularité.
"""

import os, json, pickle, logging, time
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD  # noqa: F401
import requests

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =========================================================
# CONFIG
# =========================================================
MODELS_DIR       = os.getenv("MODELS_DIR", "./models")
# ── Email config ──────────────────────────────────────────
SMTP_EMAIL    = os.getenv("SMTP_EMAIL",    "mbeldjoudi3945@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "poby ojoe bfzm jdnv")
SMTP_TO       = os.getenv("SMTP_TO",       "mbeldjoudi3945@gmail.com")
TOP_K_DEFAULT    = 10
MIN_RATING_COUNT = 2
MMR_LAMBDA       = 0.5
ALPHA_MID        = 7
ALPHA_STEEP      = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# =========================================================
# HELPERS
# =========================================================
def ok(data, meta=None):
    p = {"status": "ok", "data": data}
    if meta: p["meta"] = meta
    return jsonify(p)

def err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code

def clean_nan_for_json(v):
    """Convertit récursivement tout type non sérialisable JSON en type Python natif."""
    if isinstance(v, np.integer):   return int(v)
    if isinstance(v, np.floating):  return None if (np.isnan(v) or np.isinf(v)) else float(v)
    if isinstance(v, np.bool_):     return bool(v)
    if isinstance(v, np.ndarray):   return v.tolist()
    if isinstance(v, dict):         return {k: clean_nan_for_json(x) for k, x in v.items()}
    if isinstance(v, list):         return [clean_nan_for_json(x) for x in v]
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    try:
        if pd.isna(v): return None
    except (TypeError, ValueError):
        pass
    return v

def body_json():
    return request.get_json(silent=True) or {}

def get_int(src, *keys, default=None):
    for k in keys:
        v = src.get(k)
        if v not in (None, ""):
            try: return int(v)
            except: pass
    return default

def get_float(src, *keys, default=None):
    for k in keys:
        v = src.get(k)
        if v not in (None, ""):
            try: return float(v)
            except: pass
    return default

def get_bool(src, *keys, default=False):
    for k in keys:
        if k in src:
            v = src[k]
            if isinstance(v, bool): return v
            if isinstance(v, str):  return v.lower() in {"1","true","yes","oui"}
            return bool(v)
    return default

def segment_label(n: int) -> str:
    if n < 5:  return "Découvreur"
    if n < 15: return "Lecteur"
    return "Passionné"

# =========================================================
# CHARGEMENT
# =========================================================
def _path(f):
    p = os.path.join(MODELS_DIR, f)
    if not os.path.exists(p): raise FileNotFoundError(f"Manquant : {p}")
    return p

def load_artifacts():
    log.info("Chargement des artefacts...")
    t0 = time.time()
    with open(_path("svd_model.pkl"), "rb") as f: svd = pickle.load(f)
    edf  = pd.read_pickle(_path("book_embeddings_df.pkl"))
    emb  = np.load(_path("embedding_matrix.npy"))
    bdf  = pd.read_pickle(_path("books_base_df_final.pkl"))
    udf  = pd.read_pickle(_path("user_profiles_df_final.pkl"))
    tdf  = pd.read_pickle(_path("train_df.pkl"))
    test_df_loaded = pd.read_pickle(_path("test_df.pkl"))
    pdf  = pd.read_pickle(_path("popular_books_df.pkl"))
    with open(_path("hybrid_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    beta = float(cfg["best_beta"])
    log.info(f"OK en {time.time()-t0:.1f}s | Books:{len(bdf)} Users:{len(udf)} Emb:{emb.shape} β={beta}")
    return svd, edf, emb, bdf, udf, tdf, pdf, beta, test_df_loaded

(SVD_MODEL, EMB_DF, EMB_MATRIX, BOOKS_DF, UPROF_DF, TRAIN_DF, POP_DF, BEST_BETA, TEST_DF) = load_artifacts()

# =========================================================
# NORMALISATION
# =========================================================
def normalize_frames():
    global EMB_DF, BOOKS_DF, UPROF_DF, TRAIN_DF, POP_DF
    for n in ["EMB_DF","BOOKS_DF","UPROF_DF","TRAIN_DF","POP_DF"]:
        globals()[n] = globals()[n].copy()

    def toint(df, c):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    def tofloat(df, c):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    def ensure(df, c, default):
        if c not in df.columns: df[c] = default

    toint(EMB_DF,   "book_id_mapping")
    toint(BOOKS_DF, "book_id_mapping")
    toint(UPROF_DF, "user_id")
    toint(TRAIN_DF, "user_id")
    toint(TRAIN_DF, "book_id_mapping")
    tofloat(TRAIN_DF, "rating")

    for c in ["title","genre_clean","description"]: ensure(BOOKS_DF, c, "")
    ensure(BOOKS_DF, "rating_count",     0)
    ensure(BOOKS_DF, "rating_mean",      0.0)
    ensure(BOOKS_DF, "popularity_score", 0.0)
    ensure(BOOKS_DF, "num_pages",        0)
    ensure(EMB_DF,   "title_final",      "")
    ensure(EMB_DF,   "text_quality",     "unknown")

normalize_frames()

# =========================================================
# LOOKUPS
# =========================================================
def build_lookups():
    log.info("Construction des lookups...")
    b2m = {int(b): i for i, b in enumerate(EMB_DF["book_id_mapping"].tolist())}
    ab  = np.array(EMB_DF["book_id_mapping"].tolist(), dtype=int)

    uprof_lk = {int(r["user_id"]): np.array(r["user_profile_embedding"], dtype=np.float32)
                for _, r in UPROF_DF.iterrows()}
    ustats_lk = {int(r["user_id"]): {"count": int(r.get("ucount",0)), "mean": float(r.get("umean",0.0))}
                 for _, r in UPROF_DF.iterrows()}

    bpop_lk = {int(r["book_id_mapping"]): float(r.get("popularity_score",0.0)) for _, r in BOOKS_DF.iterrows()}
    pa = np.array(list(bpop_lk.values()), dtype=float)
    pmin, pmax = (pa.min(), pa.max()) if len(pa) else (0.0, 1.0)
    def npop(s): return float((s-pmin)/(pmax-pmin)) if pmax != pmin else 0.5
    pvec = np.array([npop(bpop_lk.get(int(b),0.0)) for b in ab], dtype=np.float32)

    book_emb_lk = {int(r["book_id_mapping"]): np.array(r["embedding"], dtype=np.float32)
                   for _, r in EMB_DF.iterrows()}
    btitle_lk = {int(r["book_id_mapping"]): str(r.get("title_final","Titre inconnu"))
                 for _, r in EMB_DF.iterrows()}
    for _, r in BOOKS_DF.iterrows():
        bid = int(r["book_id_mapping"])
        if bid not in btitle_lk: btitle_lk[bid] = str(r.get("title","Titre inconnu"))

    genre_map       = {int(k): v for k,v in BOOKS_DF.set_index("book_id_mapping")["genre_clean"].to_dict().items()}
    rc_map          = {int(k): v for k,v in BOOKS_DF.set_index("book_id_mapping")["rating_count"].to_dict().items()}
    rating_mean_map = {int(k): v for k,v in BOOKS_DF.set_index("book_id_mapping")["rating_mean"].to_dict().items()}
    desc_map        = {int(k): v for k,v in BOOKS_DF.set_index("book_id_mapping")["description"].to_dict().items()}
    pages_map       = {int(k): v for k,v in BOOKS_DF.set_index("book_id_mapping")["num_pages"].to_dict().items()}

    seen_lk  = TRAIN_DF.groupby("user_id")["book_id_mapping"].apply(set).to_dict()
    all_bids = set(int(b) for b in BOOKS_DF["book_id_mapping"].unique())
    b2idx    = {int(b): i for i, b in enumerate(EMB_DF["book_id_mapping"])}

    log.info(f"Lookups OK — {len(uprof_lk)} profils | {len(btitle_lk)} titres")
    full_hist = pd.concat([TRAIN_DF, TEST_DF], ignore_index=True)
    hist_lk   = (full_hist.groupby("user_id")
             .apply(lambda x: x[["book_id_mapping","rating"]].to_dict(orient="records"))
             .to_dict())
    return (b2m, ab, uprof_lk, ustats_lk, bpop_lk, npop, pvec,
            book_emb_lk, btitle_lk, genre_map, rc_map, rating_mean_map,
            desc_map, pages_map, seen_lk, all_bids, b2idx, hist_lk)

(B2M, AB, UPROF_LK, USTATS_LK, BPOP_LK, NPOP, PVEC,
 BOOK_EMB_LK, BTITLE_LK, GENRE_MAP, RC_MAP, RATING_MEAN_MAP,
 DESC_MAP, PAGES_MAP, SEEN_LK, ALL_BIDS, BOOKID_TO_IDX, HIST_LK) = build_lookups()

# =========================================================
# SCORING
# =========================================================
def alpha_sig(n):
    return 0.5 + 0.35 / (1 + np.exp(-ALPHA_STEEP * (n - ALPHA_MID)))

def sigmoid_norm(z):
    return 1.0 / (1.0 + np.exp(-np.array(z, dtype=float)))

def sc_pop(cands):
    return PVEC[np.array([B2M[int(b)] for b in cands])]

def sc_svd(uid, cands):
    return np.array([np.clip((SVD_MODEL.predict(int(uid),int(b)).est-1)/4,0,1) for b in cands], dtype=np.float32)

def sc_cb(uid, cands):
    if uid not in UPROF_LK: return np.zeros(len(cands), dtype=np.float32)
    uv = UPROF_LK[uid].astype(np.float32)
    idx = np.array([B2M[int(b)] for b in cands])
    return np.clip((EMB_MATRIX[idx] @ uv + 1) / 2, 0, 1)

def get_user_top_genre(uid):
    seen = SEEN_LK.get(uid, set())
    if not seen: return None
    counts = {}
    for bid in seen:
        for g in str(GENRE_MAP.get(int(bid),"")).lower().split("|"):
            g = g.strip()
            if g and g not in ("unknown",""): counts[g] = counts.get(g,0) + 1
    return max(counts, key=counts.get) if counts else None

def sc_genre_boost(uid, cands, boost=0.15):
    top = get_user_top_genre(uid)
    if not top: return np.zeros(len(cands), dtype=np.float32)
    return np.array([float(boost) if top in str(GENRE_MAP.get(int(b),"")).lower().split("|")[0].strip() else 0.0
                     for b in cands], dtype=np.float32)

def sc_hybrid(uid, cands, beta=None):
    if beta is None: beta = BEST_BETA
    cf = sc_svd(uid, cands); cb = sc_cb(uid, cands); pop = sc_pop(cands)
    gb = sc_genre_boost(uid, cands)
    cf_z  = (cf  - cf.mean())  / (cf.std()  + 1e-8)
    cb_z  = (cb  - cb.mean())  / (cb.std()  + 1e-8)
    pop_z = (pop - pop.mean()) / (pop.std() + 1e-8)
    a = alpha_sig(USTATS_LK.get(uid,{}).get("count",0))
    return a*cf_z + (1-a)*cb_z + beta*pop_z + gb

# =========================================================
# MMR
# =========================================================
def mmr_rerank(cdf, scol="score_normalise", lam=MMR_LAMBDA, top_n=TOP_K_DEFAULT, max_per_genre=4):
    if len(cdf) <= top_n: return cdf.copy()
    c = cdf.copy().reset_index(drop=True)
    bids = c["book_id_mapping"].tolist()
    dim  = EMB_MATRIX.shape[1]
    es   = np.array([BOOK_EMB_LK.get(int(b), np.zeros(dim,dtype=np.float32)) for b in bids])
    sm   = cosine_similarity(es)

    def gg(idx):
        return str(GENRE_MAP.get(int(c.loc[idx,"book_id_mapping"]),"unknown")).split("|")[0].strip().lower()

    first = int(c[scol].idxmax())
    sel   = [first]; rem = list(set(range(len(c))) - {first}); gc = {gg(first): 1}

    while len(sel) < top_n and rem:
        best_s, best_i = -np.inf, None
        for ri in rem:
            if gc.get(gg(ri), 0) >= max_per_genre: continue
            ms = max(sm[ri, si] for si in sel)
            sc_ = lam * c.loc[ri, scol] - (1-lam) * ms
            if sc_ > best_s: best_s, best_i = sc_, ri
        if best_i is None:
            for ri in rem:
                ms = max(sm[ri, si] for si in sel)
                sc_ = lam * c.loc[ri, scol] - (1-lam) * ms
                if sc_ > best_s: best_s, best_i = sc_, ri
        if best_i is not None:
            sel.append(best_i); rem.remove(best_i)
            g = gg(best_i); gc[g] = gc.get(g,0)+1
        else: break

    return c.loc[sel].reset_index(drop=True)

# =========================================================
# RECOMMANDATION
# =========================================================
def _pop_fallback(top_n, reason="Livre populaire"):
    r = POP_DF.head(top_n).copy()
    mn, mx = float(r["popularity_score"].min()), float(r["popularity_score"].max())
    r["score_normalise"]     = ((r["popularity_score"]-mn)/(mx-mn+1e-8)).round(4)
    r["recommendation_type"] = "cold_start_popularity"
    r["reason"]              = reason
    cols = ["book_id_mapping","book_id","title","score_normalise","rating_count","rating_mean","recommendation_type","reason"]
    return clean_nan_for_json(r[[c for c in cols if c in r.columns]].to_dict(orient="records"))

def recommend_known_user(uid, top_n=TOP_K_DEFAULT, min_rc=MIN_RATING_COUNT,
                         genre_filter=None, min_score=0.0, diversity=True):
    raw = list(ALL_BIDS - SEEN_LK.get(uid, set()))
    if min_rc > 0: raw = [b for b in raw if RC_MAP.get(int(b),0) >= min_rc]
    if genre_filter:
        gf = genre_filter.lower().strip()
        raw = [b for b in raw if gf in str(GENRE_MAP.get(int(b),"")).lower()]
    cands = np.array([b for b in raw if int(b) in B2M], dtype=int)
    if len(cands) == 0: return _pop_fallback(top_n)

    scores  = sc_hybrid(uid, cands)
    pre_k   = min(top_n*3, len(cands))
    top_idx = np.argsort(scores)[::-1][:pre_k]
    top_c, top_z = cands[top_idx], scores[top_idx]
    top_sn  = sigmoid_norm(top_z)

    hist = TRAIN_DF[TRAIN_DF["user_id"]==uid].merge(
        BOOKS_DF[["book_id_mapping","title"]], on="book_id_mapping", how="left"
    ).sort_values("rating", ascending=False)
    fav       = hist[hist["rating"]>=4.0]["title"].tolist()
    top_genre = get_user_top_genre(uid)
    top3      = list(dict.fromkeys(fav[:5]))[:3] if fav else []
    ref_cycle = 0

    rows = []
    for b, sz, sn in zip(top_c, top_z, top_sn):
        genre_livre = str(GENRE_MAP.get(int(b), "")).lower().split("|")[0].strip()
        if top_genre and top_genre in genre_livre:
            reason = f"Correspond à votre goût pour le {top_genre}"
        elif top3:
            reason = f"Parce que vous avez aimé {str(top3[ref_cycle % len(top3)])[:55]}"
            ref_cycle += 1
        else:
            reason = "Recommandé selon votre profil de lecture"
        rows.append({
            "user_id": int(uid), "book_id_mapping": int(b),
            "title": BTITLE_LK.get(int(b),"?"),
            "score_normalise": round(float(sn),4), "z_score": round(float(sz),4),
            "genre": str(GENRE_MAP.get(int(b),"")),
            "rating_count": int(RC_MAP.get(int(b),0)),
            "rating_mean": float(RATING_MEAN_MAP.get(int(b),0.0)),
            "recommendation_type": "hybrid_known_user",
            "reason": reason
        })
   

    df = pd.DataFrame(rows)
    if min_score > 0: df = df[df["score_normalise"] >= min_score]
    result = mmr_rerank(df, "score_normalise", MMR_LAMBDA, top_n) if diversity else df.head(top_n)
    return clean_nan_for_json(result.to_dict(orient="records"))

def recommend_new_user(liked_ids, top_n=TOP_K_DEFAULT):
    vecs = [BOOK_EMB_LK[b] for b in liked_ids if b in BOOK_EMB_LK]
    if not vecs: return _pop_fallback(top_n)
    prof = np.mean(vecs, axis=0).astype(np.float32)

    # Cycle sur les 3 titres aimés pour varier les raisons
    liked_titles = [BTITLE_LK.get(b,"?")[:55] for b in liked_ids if b in BTITLE_LK][:3]

    # Tous les genres détectés — pas juste le dominant
    liked_genres = []
    for b in liked_ids:
        g = str(GENRE_MAP.get(int(b),"")).split("|")[0].strip().lower()
        if g and g != "unknown": liked_genres.append(g)
    dominant_genres = set(liked_genres) if liked_genres else set()

    rows = []
    for bid in ALL_BIDS:
        if int(bid) in {int(x) for x in liked_ids} or bid not in BOOK_EMB_LK: continue
        bv  = BOOK_EMB_LK[bid]
        raw = float(np.dot(prof,bv)/(np.linalg.norm(prof)*np.linalg.norm(bv)+1e-8))
        cb  = float(np.clip((raw+1)/2,0,1))
        pop = float(NPOP(BPOP_LK.get(int(bid),0.0)))
        idx = len(rows) % len(liked_titles)
        reason = f"Parce que vous aimez {liked_titles[idx]}" if liked_titles else "Selon vos préférences"

        # Boost si le livre appartient à N'IMPORTE LEQUEL des genres sélectionnés
        genre_boost = 0.0
        if dominant_genres:
            bid_genre = str(GENRE_MAP.get(int(bid),"")).split("|")[0].strip().lower()
            if bid_genre in dominant_genres:
                genre_boost = 0.15

        rows.append({
            "book_id_mapping": int(bid),
            "title":           BTITLE_LK.get(int(bid),"?"),
            "score_normalise": round(float(min(0.7*cb + 0.3*pop + genre_boost, 1.0)), 4),
            "cb_score":        round(cb, 4),
            "pop_score":       round(pop, 4),
            "genre":           str(GENRE_MAP.get(int(bid),"")),
            "rating_count":    int(RC_MAP.get(int(bid),0)),
            "rating_mean":     float(RATING_MEAN_MAP.get(int(bid),0.0)),
            "recommendation_type": "content_new_user",
            "reason": reason
        })

    df = pd.DataFrame(rows).sort_values("score_normalise", ascending=False)
    result = df.head(top_n)
    return clean_nan_for_json(result.to_dict(orient="records"))
def recommend_similar(book_id_mapping=None, title=None, top_n=TOP_K_DEFAULT):
    if book_id_mapping is None:
        if title is None: raise ValueError("Fournir book_id_mapping ou title.")
        q = title.lower().strip()
        m = EMB_DF[EMB_DF["title_final"].str.lower().str.contains(q, na=False)]
        if len(m) == 0: raise ValueError(f"Aucun livre trouvé pour '{title}'.")
        pops = {int(r["book_id_mapping"]): BPOP_LK.get(int(r["book_id_mapping"]),0) for _,r in m.iterrows()}
        book_id_mapping = max(pops, key=pops.get)
    if book_id_mapping not in BOOKID_TO_IDX:
        raise ValueError(f"ID {book_id_mapping} absent de la matrice.")
    idx  = BOOKID_TO_IDX[book_id_mapping]
    sims = EMB_MATRIX @ EMB_MATRIX[idx]
    results = []
    for i in np.argsort(sims)[::-1]:
        bid = int(EMB_DF.iloc[i]["book_id_mapping"])
        if bid == book_id_mapping: continue
        results.append({"book_id_mapping": int(bid), "title": BTITLE_LK.get(bid,"?"),
                        "similarity": round(float(sims[i]),4), "genre": str(GENRE_MAP.get(bid,"")),
                        "rating_count": int(RC_MAP.get(bid,0)), "rating_mean": float(RATING_MEAN_MAP.get(bid,0.0)),
                        "reason": "Livre proche en termes de contenu"})
        if len(results) >= top_n: break
    return clean_nan_for_json(results)

def recommend_cold_start(top_n=TOP_K_DEFAULT, genre_filter=None):
    df = POP_DF.copy()
    if genre_filter:
        gf = genre_filter.lower().strip()
        df = df[df["book_id_mapping"].map(
            lambda b: str(GENRE_MAP.get(int(b),"")).lower().split("|")[0].strip() == gf
        )]
    if len(df) == 0: df = POP_DF.copy()
    top = df.head(top_n).copy()
    mn, mx = float(top["popularity_score"].min()), float(top["popularity_score"].max())
    top["score_normalise"]     = ((top["popularity_score"]-mn)/(mx-mn+1e-8)).round(4)
    top["recommendation_type"] = "cold_start_popularity"
    top["reason"]              = "Livre populaire"
    top["genre"]               = top["book_id_mapping"].map(lambda b: str(GENRE_MAP.get(int(b),"")) )
    cols = ["book_id_mapping","book_id","title","score_normalise","rating_count","rating_mean","genre","recommendation_type","reason"]
    return clean_nan_for_json(top[[c for c in cols if c in top.columns]].to_dict(orient="records"))

# =========================================================
# COUVERTURES SVG
# =========================================================
GENRE_PALETTES = {
    "fantasy":     ("#1e3a5f","#2d6a9f","#ffffff"),
    "fiction":     ("#1a4731","#2d8a5e","#ffffff"),
    "romance":     ("#7b1d3a","#c94070","#ffffff"),
    "mystery":     ("#1a1a2e","#3d5a80","#e0e0e0"),
    "thriller":    ("#0d1117","#c0392b","#ffffff"),
    "young":       ("#4a1580","#8e44ad","#ffffff"),
    "non-fiction": ("#0c3547","#1a7a9e","#ffffff"),
    "history":     ("#4a2000","#c0651a","#ffffff"),
    "children":    ("#1a4a8a","#e8940a","#ffffff"),
    "science":     ("#071426","#2980b9","#e8f4f8"),
    "poetry":      ("#2d1b5e","#7d5fc0","#ffffff"),
    "horror":      ("#0a0a0a","#cc0000","#eeeeee"),
}

def build_cover_svg(title, genre):
    gl = genre.lower()
    bg1, bg2, tc = "#1a3a6b", "#2d5fa8", "#ffffff"
    for k, pal in GENRE_PALETTES.items():
        if k in gl: bg1, bg2, tc = pal; break

    words, lines, cur = title.split(), [], ""
    for w in words:
        t = (cur+" "+w).strip()
        if len(t) > 14:
            if cur: lines.append(cur.strip())
            cur = w
        else: cur = t
        if len(lines) >= 4: break
    if cur and len(lines) < 4: lines.append(cur.strip())
    lines = lines[:4]

    mono = title[0].upper() if title else "B"
    fs   = 15 if len(lines) >= 3 else 17
    lh   = fs + 7
    y0   = 155 - (len(lines)*lh)//2

    svg_lines = ""
    for i, ln in enumerate(lines):
        y = y0 + i*lh
        safe = ln.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        svg_lines += f'<text x="100" y="{y}" text-anchor="middle" font-family="Georgia,serif" font-size="{fs}" font-weight="bold" fill="{tc}" opacity="0.96">{safe}</text>'

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="300" viewBox="0 0 200 300">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{bg1}"/><stop offset="100%" style="stop-color:{bg2}"/>
    </linearGradient>
    <pattern id="dots" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
      <circle cx="10" cy="10" r="1" fill="{tc}" opacity="0.07"/>
    </pattern>
  </defs>
  <rect width="200" height="300" fill="url(#bg)"/>
  <rect width="200" height="300" fill="url(#dots)"/>
  <rect x="0" y="0" width="7" height="300" fill="{tc}" opacity="0.20"/>
  <rect x="11" y="14" width="178" height="272" fill="none" stroke="{tc}" stroke-width="1" opacity="0.12"/>
  <text x="100" y="96" text-anchor="middle" font-family="Georgia,serif" font-size="58" font-weight="bold" fill="{tc}" opacity="0.09">{mono}</text>
  {svg_lines}
  <line x1="26" y1="248" x2="174" y2="248" stroke="{tc}" stroke-width="1" opacity="0.22"/>
  <text x="100" y="270" text-anchor="middle" font-family="Georgia,serif" font-size="10" fill="{tc}" opacity="0.55">BooksMatch</text>
</svg>"""

# =========================================================
# ROUTES
# =========================================================
@app.route("/", methods=["GET"])
def index(): return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return ok({"models_loaded": True, "books_count": int(len(BOOKS_DF)),
               "users_count": int(len(UPROF_DF)), "best_beta": float(BEST_BETA),
               "embedding_dim": int(EMB_MATRIX.shape[1])})

# APRÈS
IMG_MAP = {
    int(r["book_id_mapping"]): str(r["image_url"])
    for _, r in BOOKS_DF.iterrows()
    if pd.notna(r.get("image_url")) and str(r.get("image_url","")).strip() != ""
}

@app.route("/api/covers/<int:bid>", methods=["GET"])
def book_cover(bid):
    img_url = IMG_MAP.get(bid)
    if img_url:
        try:
            r = requests.get(img_url, timeout=5, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.goodreads.com/"
            })
            if r.status_code == 200:
                content_type = r.headers.get("Content-Type", "image/jpeg")
                return Response(r.content, mimetype=content_type)
        except Exception:
            pass
    return Response(
        build_cover_svg(BTITLE_LK.get(bid, "Livre"), str(GENRE_MAP.get(bid, ""))),
        mimetype="image/svg+xml"
    )

@app.route("/api/recommend/user", methods=["POST"])
def route_known_user():
    body = body_json()
    uid  = get_int(body, "user_id")
    if uid is None: return err("'user_id' requis.")
    if uid not in USTATS_LK: return err(f"Utilisateur {uid} inconnu.", 404)
    recos = recommend_known_user(uid, get_int(body,"top_n","tn",default=TOP_K_DEFAULT),
                                 get_int(body,"min_rating_count",default=MIN_RATING_COUNT),
                                 body.get("genre_filter") or None, get_float(body,"min_score",default=0.0),
                                 get_bool(body,"diversity",default=True))
    n_int = int(USTATS_LK.get(uid,{}).get("count",0))
    return ok(recos, meta={"user_id": int(uid), "n_interactions": n_int,
                            "alpha": round(alpha_sig(n_int),3), "segment": segment_label(n_int),
                            "strategy": "hybrid_svd_cb_pop", "top_n": len(recos)})

@app.route("/api/recommend/new", methods=["POST"])
def route_new_user():
    body = body_json()
    lids = body.get("liked_ids")
    if not lids: return err("'liked_ids' requis.")
    try: lids = [int(x) for x in lids]
    except: return err("'liked_ids' doit être une liste d'entiers.")
    recos = recommend_new_user(lids, get_int(body,"top_n","tn",default=TOP_K_DEFAULT))
    return ok(recos, meta={"liked_ids": lids, "liked_titles": [BTITLE_LK.get(b,str(b)) for b in lids],
                            "strategy": "content_based_new_user", "top_n": len(recos)})

@app.route("/api/recommend/popular", methods=["GET"])
def route_popular():
    args  = request.args
    recos = recommend_cold_start(top_n=get_int(args,"top_n","tn",default=TOP_K_DEFAULT),
                                 genre_filter=args.get("genre","").strip() or None)
    return ok(recos, meta={"strategy": "cold_start_popularity", "top_n": len(recos)})

@app.route("/api/recommend/similar", methods=["POST"])
def route_similar():
    body = body_json()
    bid  = body.get("book_id_mapping")
    ttl  = body.get("title")
    if bid is None and not ttl: return err("Fournir 'book_id_mapping' ou 'title'.")
    try:
        if bid is not None: bid = int(bid)
        recos = recommend_similar(book_id_mapping=bid, title=ttl,
                                  top_n=get_int(body,"top_n","tn",default=TOP_K_DEFAULT))
        return ok(recos, meta={"strategy": "similar_books", "top_n": len(recos)})
    except ValueError as e: return err(str(e), 404)

@app.route("/api/books", methods=["GET"])
def route_books():
    args  = request.args
    top_n = get_int(args,"top_n","tn",default=200)
    genre = args.get("genre","").strip().lower()
    df    = BOOKS_DF.copy()
    if genre: df = df[df["genre_clean"].astype(str).str.lower().str.contains(genre, na=False)]
    sc    = "popularity_score" if "popularity_score" in df.columns else "rating_count"
    df    = df.sort_values([sc,"rating_count"], ascending=False).head(top_n)
    data  = []
    for _, r in df.iterrows():
        bid = int(r["book_id_mapping"])
        data.append({"book_id_mapping": bid, "book_id": clean_nan_for_json(r.get("book_id")),
                     "title": BTITLE_LK.get(bid, str(r.get("title","Titre inconnu"))),
                     "author": str(r.get("author","") or ""),
                     "genre_clean": str(r.get("genre_clean","")),
                     "rating_count": int(r.get("rating_count",0) or 0),
                     "rating_mean": round(float(r.get("rating_mean",0.0) or 0.0),2),
                     "popularity_score": round(float(r.get("popularity_score",0.0) or 0.0),4),
                     "num_pages": clean_nan_for_json(r.get("num_pages",0)),
                     "description": str(r.get("description","") or "")})
    return ok(clean_nan_for_json(data), meta={"top_n": len(data), "genre_filter": genre or None})

@app.route("/api/books/<int:bid>", methods=["GET"])
def route_book_detail(bid):
    row = BOOKS_DF[BOOKS_DF["book_id_mapping"] == bid]
    if len(row) == 0: return err(f"Livre {bid} introuvable.", 404)
    r = row.iloc[0]
    # Forcer TOUS les types en Python natif pour éviter int32 non sérialisable
    try:
        np_ = int(r.get("num_pages",0)) if not pd.isna(r.get("num_pages",0)) else None
    except: np_ = None
    payload = {
        "book_id_mapping":  int(bid),
        "book_id":          clean_nan_for_json(r.get("book_id")),
        "title":            BTITLE_LK.get(int(bid), str(r.get("title","Titre inconnu"))),
        "author":           str(r.get("author","") or ""),
        "genre_clean":      str(r.get("genre_clean","") or ""),
        "rating_count":     int(r.get("rating_count",0) or 0),
        "rating_mean":      round(float(r.get("rating_mean",0.0) or 0.0),2),
        "popularity_score": round(float(r.get("popularity_score",0.0) or 0.0),4),
        "description":      str(r.get("description","") or ""),
        "num_pages":        np_,
    }
    return ok(clean_nan_for_json(payload))

@app.route("/api/users/<int:uid>", methods=["GET"])
def route_user_profile(uid):
    if uid not in USTATS_LK: return err(f"Utilisateur {uid} introuvable.", 404)
    n_int = int(USTATS_LK[uid]["count"])
    hist_records = HIST_LK.get(uid, [])
    hist = (pd.DataFrame(hist_records)
             .merge(BOOKS_DF[["book_id_mapping","title","genre_clean"]], on="book_id_mapping", how="left")
             .sort_values("rating", ascending=False).head(50)
             [["book_id_mapping","title","genre_clean","rating"]].fillna(""))
    return ok({"user_id": int(uid), "n_interactions": n_int,
               "mean_rating": round(float(USTATS_LK[uid].get("mean",0.0)),2),
               "alpha": round(alpha_sig(n_int),3), "segment": segment_label(n_int),
               "history": clean_nan_for_json(hist.to_dict(orient="records"))})



@app.route("/api/contact", methods=["POST"])
def route_contact():
    body = body_json()
    name    = str(body.get("name",    "")).strip()
    email   = str(body.get("email",   "")).strip()
    subject = str(body.get("subject", "")).strip()
    message = str(body.get("message", "")).strip()

    if not name or not email or not message:
        return err("Tous les champs sont requis.")

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[BooksMatch] {subject or 'Question'} — {name}"
        msg["From"]    = SMTP_EMAIL
        msg["To"]      = SMTP_TO

        html = f"""
        <div style="font-family:sans-serif;max-width:600px;margin:0 auto;">
          <div style="background:#1a3a6b;padding:24px;border-radius:12px 12px 0 0;">
            <h2 style="color:#fff;margin:0;">📚 BooksMatch — Nouvelle question</h2>
          </div>
          <div style="background:#f5f3ee;padding:24px;border-radius:0 0 12px 12px;">
            <table style="width:100%;border-collapse:collapse;">
              <tr><td style="padding:8px 0;color:#5c6b7a;font-size:13px;">Nom</td>
                  <td style="padding:8px 0;font-weight:700;color:#0f1923;">{name}</td></tr>
              <tr><td style="padding:8px 0;color:#5c6b7a;font-size:13px;">Email</td>
                  <td style="padding:8px 0;font-weight:700;color:#0f1923;">
                    <a href="mailto:{email}" style="color:#1a3a6b;">{email}</a></td></tr>
              <tr><td style="padding:8px 0;color:#5c6b7a;font-size:13px;">Sujet</td>
                  <td style="padding:8px 0;font-weight:700;color:#0f1923;">{subject}</td></tr>
            </table>
            <div style="margin-top:18px;background:#fff;border-left:4px solid #e8640a;
                        border-radius:8px;padding:16px;">
              <p style="color:#5c6b7a;font-size:12px;margin:0 0 8px;">Message :</p>
              <p style="color:#0f1923;line-height:1.7;margin:0;">{message.replace(chr(10),'<br>')}</p>
            </div>
            <p style="margin-top:18px;color:#94a3b8;font-size:11px;">
              Envoyé depuis BooksMatch — IFM30542-H2026 Groupe 7
            </p>
          </div>
        </div>
        """

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, SMTP_TO, msg.as_string())

        log.info(f"Email envoyé depuis {email} — sujet: {subject}")
        return ok({"sent": True})

    except Exception as e:
        log.error(f"Erreur email : {e}")
        return err("Impossible d'envoyer le message. Réessayez plus tard.", 500)



# =========================================================
# ERREURS
# =========================================================
@app.errorhandler(404)
def not_found(_): return err("Ressource introuvable.", 404)

@app.errorhandler(500)
def server_error(e):
    log.error(f"Erreur 500 : {e}")
    return err("Erreur interne du serveur.", 500)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    host  = os.getenv("FLASK_HOST",  "127.0.0.1")
    port  = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    log.info(f"Interface : http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug)