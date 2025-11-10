from __future__ import annotations
import os, re, ast, argparse, math, json, unicodedata, string, random, time, sys, subprocess, webbrowser
from typing import List, Dict, Tuple, Optional
import warnings

# Répertoire de sortie principal pour W2V
BASE_W2V_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "w2v_out")

def _save_params(outdir: str, params: dict, metrics: dict):
    """
    Sauvegarde les hyperparamètres et métriques dans un fichier params_metrics.json.
    """
    d = {"params": params, "metrics": metrics}
    ensure_dir(outdir)
    fp = os.path.join(outdir, "params_metrics.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"Paramètres et métriques sauvegardés → {os.path.abspath(fp)}")

import pandas as pd
import numpy as np

# gensim / nltk / sklearn / matplotlib
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# utilitaires 
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

# Helper: open file/folder in OS default app
def open_file_in_os(path: str) -> None:
    """Ouvre un fichier/dossier avec l'app par défaut du système (macOS/Windows/Linux)."""
    try:
        if sys.platform == "darwin":
            # macOS
            subprocess.run(["open", path], check=False)
        elif os.name == "nt":
            # Windows
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            # Linux
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        try:
            webbrowser.open("file://" + os.path.abspath(path))
        except Exception:
            pass

def deaccent(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def norm_text(s: str) -> str:
    s = deaccent(s or "")
    s = s.lower()
    # garde lettres/chiffres/espaces de manière simple
    s = re.sub(r"[^a-z0-9'\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_parse_list_of_dicts(val: str) -> List[str]:
    """keywords/genres sont des listes de dicts sous forme string; renvoie la liste des 'name' en minuscules."""
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        obj = ast.literal_eval(val)
        out = []
        for x in obj:
            if isinstance(x, dict) and "name" in x and isinstance(x["name"], str):
                out.append(norm_text(x["name"]))
        return out
    except Exception:
        return []

def find_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p): return p
    return None

def tokenize_sentences(doc: str, lang_stops) -> List[str]:
    toks = word_tokenize(doc)
    stops = set(lang_stops)
    # Add punctuation to stopwords
    stops |= set(string.punctuation)
    out = []
    for t in toks:
        tt = t.strip("'")
        if not tt:
            continue
        if tt.lower() in stops:
            continue
        if tt.isnumeric() and len(tt) <= 2:
            continue
        # Remove short tokens except "ai"
        if len(tt) < 3 and tt.lower() != "ai":
            continue
        # Remove stopwords and punctuation again (for safety)
        if tt.lower() in stops:
            continue
        out.append(tt)
    # Remove generic tokens
    generic_tokens = {
        # very generic function words / fillers
        "one","time","world","people","story","stories","character","characters","life","lives",
        "man","woman","men","women","new","young","old","day","days","year","years",
        "go","goes","going","make","makes","made","take","takes","taken","come","comes","coming","get","gets","got",
        "also","however","well","first","two","three","series","feature","features","documentary"
        # NOTE: do NOT include these to keep them in vocab for analysis tabs:
        # 'space','love','war','family','police','friendship','robot'
        # Keep high-level genres too; we will prune them by doc-frequency instead of hard-removing here.
    }
    out = [t for t in out if t.lower() not in generic_tokens]
    return out

# --- Lemmatisation légère avec NLTK ---
def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Applique une lemmatisation légère sur une liste de tokens (WordNetLemmatizer, mode 'n').
    """
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t, pos="n") for t in tokens]

# === Pruning helpers: docfreq-based token pruning ===
def _docfreq(tokens_list: List[List[str]]) -> Dict[str, int]:
    from collections import Counter
    df = Counter()
    for toks in tokens_list:
        df.update(set(toks))
    return df

def prune_by_docfreq(tokens_list: List[List[str]], max_df_ratio: float = 0.35, min_df: int = 3, extra_stop: Optional[List[str]] = None, keep: Optional[set] = None) -> Tuple[List[List[str]], Dict[str, int], set]:
    """
    Supprime:
      - les tokens apparaissant dans plus de max_df_ratio des documents,
      - les tokens apparaissant dans moins de min_df documents,
      - et les tokens listés dans extra_stop (optionnel).
    Renvoie (tokens_pruned, docfreq_dict, removed_set).
    """
    N = max(1, len(tokens_list))
    df = _docfreq(tokens_list)
    removed = {t for t, c in df.items() if (c / N) > max_df_ratio or c < min_df}
    # Always keep specific tokens (e.g., probe words like 'war','family',...)
    if keep:
        removed -= set(keep)
    if extra_stop:
        removed |= set(extra_stop)
    pruned = [[t for t in toks if t not in removed] for toks in tokens_list]
    return pruned, df, removed

def mean_pool(tokens: List[str], model: Word2Vec, idf: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
    vecs = []
    weights = []
    kv = model.wv
    for w in tokens:
        if w in kv:
            vecs.append(kv[w])
            weights.append((idf.get(w, 1.0) if idf else 1.0))
    if not vecs:
        return None
    V = np.vstack(vecs)
    w = np.array(weights, dtype=float)
    sw = w.sum()
    if sw <= 0:
        return V.mean(axis=0)
    return (V * w[:, None]).sum(axis=0) / sw

# entropie corpus 
def compute_corpus_entropy(sentences: List[List[str]]) -> float:
    """Calcule l'entropie empirique du corpus sur la distribution des tokens."""
    from collections import Counter
    tokens = [t for s in sentences for t in s]
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)

# charge & prépare 
def load_corpus(data_dir: str, phrases: bool = True) -> Tuple[List[List[str]], pd.DataFrame]:
    """
    Renvoie:
      - sentences: List[List[str]] (phrases/tokens par film)
      - df: DataFrame avec id, title, tokens, genres_list
    Utilise movies_metadata.csv (+ keywords.csv si dispo).
    """
    # Use archive folder on Desktop for CSVs
    archive_dir = os.path.expanduser("~/Desktop/archive")
    movies_paths = [
        os.path.join(archive_dir, "movies_metadata.csv"),
        os.path.join(data_dir, "movies_metadata.csv"),
        "movies_metadata.csv"
    ]
    keywords_paths = [
        os.path.join(archive_dir, "keywords.csv"),
        os.path.join(data_dir, "keywords.csv"),
        "keywords.csv"
    ]

    movies_csv = find_existing(movies_paths)
    if not movies_csv:
        raise SystemExit("movies_metadata.csv introuvable — passe --data-dir /chemin/vers/csv")
    kw_csv = find_existing(keywords_paths)  # optionnel

    # lecture movies; les types de colonnes sont hétérogènes, on sécurise
    df = pd.read_csv(movies_csv, low_memory=False)
    # garde colonnes d'intérêt
    for col in ["id", "title", "overview", "tagline", "genres", "original_language"]:
        if col not in df.columns:
            df[col] = np.nan

    # id propre (string) → certaines lignes sont bizarres
    df["id"] = df["id"].astype(str).str.extract(r"(\d+)")[0]
    df = df.dropna(subset=["id"]).copy()

    # parse genres
    df["genres_list"] = df["genres"].apply(safe_parse_list_of_dicts)

    # ajoute keywords si dispo
    if kw_csv:
        kw = pd.read_csv(kw_csv)
        kw["id"] = kw["id"].astype(str)
        kw["kw_list"] = kw["keywords"].apply(safe_parse_list_of_dicts)
        df = df.merge(kw[["id","kw_list"]], on="id", how="left")
    else:
        df["kw_list"] = [[] for _ in range(len(df))]

    # texte de base
    df["overview"] = df["overview"].fillna("").astype(str)
    df["tagline"]  = df["tagline"].fillna("").astype(str)
    df["overview"] = df["overview"].replace("nan", "")
    df["tagline"]  = df["tagline"].replace("nan", "")

    # normalisation
    df["text"] = (df["overview"] + " " + df["tagline"]).map(norm_text)

    # stopwords EN+FR
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    stops = set(stopwords.words("english"))
    try:
        stops |= set(stopwords.words("french"))
    except Exception:
        pass

    # Filtrage pour ne garder que les films en anglais
    df = df[df["original_language"] == "en"].copy()

    # tokenise
    df["tokens"] = df["text"].apply(lambda s: tokenize_sentences(s, stops))

    # --- Lemmatisation légère ---
    df["tokens"] = df["tokens"].apply(lemmatize_tokens)

    # ajoute genres/keywords comme tags (mots composés déjà normalisés)
    def extend_tokens(row):
        toks = []
        # sécurité : tokens
        if isinstance(row.get("tokens"), list):
            toks = list(row["tokens"])
        # sécurité : genres
        genres = row.get("genres_list", [])
        if isinstance(genres, list):
            toks += [g.replace(" ", "_") for g in genres]
        # sécurité : keywords
        kw = row.get("kw_list", [])
        if not isinstance(kw, list):
            kw = []
        toks += [k.replace(" ", "_") for k in kw]
        return toks

    df["tokens"] = df.apply(extend_tokens, axis=1)

    sentences = df["tokens"].tolist()

    if phrases:
        try:
            phrases_big = Phrases(sentences, min_count=20, threshold=10)
            phraser_big = Phraser(phrases_big)
            sentences = [phraser_big[s] for s in sentences]

            phrases_tri = Phrases(sentences, min_count=30, threshold=12)
            phraser_tri = Phraser(phrases_tri)
            sentences = [phraser_tri[s] for s in sentences]
        except Exception:
            pass

    df["tokens"] = [list(s) for s in sentences]
    df["tokens_raw"] = [list(s) for s in sentences]

    return sentences, df[["id","title","tokens","genres_list","tokens_raw"]].reset_index(drop=True)

# entraînement 
def train_word2vec(
    sentences: List[List[str]],
    outdir: str,
    seed: int = 42,
    vector_size=120,
    window=10,
    min_count=3,
    epochs=10
) -> Word2Vec:
    random.seed(seed)
    np.random.seed(seed)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=os.cpu_count() or 4,
        sg=1,               # skip-gram
        negative=15,
        sample=1e-5,
        epochs=epochs,
        seed=seed
    )
    ensure_dir(outdir)
    model.wv.save(os.path.join(outdir, "w2v_movies.kv"))
    return model

# évaluation qualitative 
def print_neighbors(model: Word2Vec, words: List[str], k: int = 10):
    print("\n=== Voisins les plus proches ===")
    top_freq = set(list(model.wv.key_to_index.keys())[:200])
    for w in words:
        if w not in model.wv:
            print(f"{w:>12s} → [hors vocabulaire]")
            continue
        candidates = model.wv.most_similar(w, topn=200)
        clean = []
        for nb, sc in candidates:
            if nb == w or nb in top_freq:
                continue
            if nb.isnumeric() or len(nb) < 3:
                continue
            clean.append((nb, sc))
            if len(clean) >= 10:
                break
        if clean:
            print(f"{w:>12s} → {[f'{a}({b:.3f})' for a,b in clean]}")
        else:
            print(f"{w:>12s} → [aucun voisin exploitable]")

def build_movie_embeddings(df: pd.DataFrame, model: Word2Vec, idf: Optional[Dict[str, float]] = None, tokens_col: str = "tokens") -> pd.DataFrame:
    embs = []
    for _, row in df.iterrows():
        toks = row[tokens_col] if tokens_col in row and isinstance(row[tokens_col], list) else row.get("tokens", [])
        v = mean_pool(toks or [], model, idf=idf)
        embs.append(v)
    M = np.vstack([e if e is not None else np.zeros(model.vector_size) for e in embs])
    out = df.copy()
    out["has_vec"] = [e is not None for e in embs]
    out["emb"] = list(M)
    return out

def similar_movies(df_emb: pd.DataFrame, title: str, topk: int = 10) -> List[Tuple[str,float]]:
    # cosine vs titres (moyenne des tokens)
    row = df_emb[df_emb.title.str.lower()==title.lower()]
    if row.empty or not row.iloc[0]["has_vec"]:
        return []
    v = row.iloc[0]["emb"]
    M = np.vstack(df_emb["emb"].values)
    denom = (np.linalg.norm(M, axis=1) * (np.linalg.norm(v)+1e-9)) + 1e-9
    sims = (M @ v) / denom
    idx = sims.argsort()[::-1]
    out = []
    for j in idx:
        if df_emb.iloc[j]["title"].lower() == title.lower(): 
            continue
        out.append((df_emb.iloc[j]["title"], float(sims[j])))
        if len(out) >= topk: break
    return out

def neighbor_genre_agreement(df_emb: pd.DataFrame, k: int = 1) -> float:
    """% de films dont le plus proche partage ≥1 genre (sanity check simple)."""
    dfv = df_emb[df_emb["has_vec"]].reset_index(drop=True)
    if len(dfv) < 10: 
        return 0.0
    M = np.vstack(dfv["emb"].values)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    sims = (M @ M.T) / (norms @ norms.T)
    np.fill_diagonal(sims, -1.0)
    hits = 0
    for i in range(len(dfv)):
        j = int(np.argmax(sims[i]))
        gi = set(dfv.iloc[i]["genres_list"])
        gj = set(dfv.iloc[j]["genres_list"])
        if gi and gj and (gi & gj):
            hits += 1
    return hits / len(dfv)

#  visus 
def tsne_plot_words(model: Word2Vec, outdir: str, topn: int = 200):
    # prend les top mots par fréquence
    vocab = list(model.wv.key_to_index.keys())[:topn]
    X = np.array([model.wv[w] for w in vocab])
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=42)
    Z = tsne.fit_transform(X)
    plt.figure(figsize=(10,8))
    plt.scatter(Z[:,0], Z[:,1], s=8)
    for i, w in enumerate(vocab):
        if i % max(1, topn//100) == 0:
            plt.text(Z[i,0], Z[i,1], w, fontsize=8)
    plt.title("Word2Vec — t-SNE (échantillon vocab)")
    ensure_dir(outdir)
    fp = os.path.join(outdir, "tsne_words.png")
    plt.tight_layout(); plt.savefig(fp, dpi=160); plt.close()
    print("Figure t-SNE sauvegardée →", os.path.abspath(fp))

# =========================
# ÉVALUATION EXTRINSÈQUE (Genres)
# =========================

def _prepare_multilabel(meta: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, MultiLabelBinarizer]:
    """Filtre les films avec genres non vides et renvoie (df, Y, mlb)."""
    df = meta.copy()
    df = df[df["genres_list"].apply(lambda g: isinstance(g, list) and len(g) > 0)].reset_index(drop=True)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["genres_list"])
    return df, Y, mlb

def eval_tfidf_baseline(meta: pd.DataFrame, outdir: str) -> Dict[str, float]:
    """Baseline TF-IDF + OneVsRest(LogReg) pour prédire les genres."""
    df, Y, mlb = _prepare_multilabel(meta)
    texts = [" ".join(toks) if isinstance(toks, list) else "" for toks in df["tokens"]]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear"))
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    micro = f1_score(yte, yhat, average="micro")
    macro = f1_score(yte, yhat, average="macro")
    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, "eval_tfidf.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"micro_f1": micro, "macro_f1": macro, "n_samples": int(X.shape[0]), "n_labels": int(Y.shape[1])}, f, indent=2)
    print(f"\n[TF-IDF] micro-F1={micro:.3f} | macro-F1={macro:.3f}  → {out_json}")
    return {"micro_f1": micro, "macro_f1": macro}

def eval_w2v_classifier(df_emb: pd.DataFrame, outdir: str) -> Dict[str, float]:
    """LogReg multi-étiquette sur embeddings W2V (moyenne des tokens)."""
    dfv = df_emb[df_emb["has_vec"]].reset_index(drop=True)
    dfv = dfv[dfv["genres_list"].apply(lambda g: isinstance(g, list) and len(g) > 0)].reset_index(drop=True)
    if len(dfv) < 50:
        print("[W2V] Trop peu d'exemples avec vecteur pour évaluer.")
        return {"micro_f1": float("nan"), "macro_f1": float("nan")}
    M = np.vstack(dfv["emb"].values)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(dfv["genres_list"])
    Xtr, Xte, ytr, yte = train_test_split(M, Y, test_size=0.2, random_state=42)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear"))
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    micro = f1_score(yte, yhat, average="micro")
    macro = f1_score(yte, yhat, average="macro")
    out_json = os.path.join(outdir, "eval_w2v.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"micro_f1": micro, "macro_f1": macro, "n_samples": int(M.shape[0]), "n_labels": int(Y.shape[1])}, f, indent=2)
    print(f"[W2V ] micro-F1={micro:.3f} | macro-F1={macro:.3f}  → {out_json}")
    return {"micro_f1": micro, "macro_f1": macro}

def train_fasttext_and_eval(sentences: List[List[str]], meta: pd.DataFrame, outdir: str) -> Optional[Dict[str, float]]:
    """Optionnel: entraîne FastText (gensim) puis évalue comme W2V."""
    try:
        from gensim.models.fasttext import FastText
    except Exception:
        print("[FastText] Non disponible (gensim FastText introuvable).")
        return None
    print(" Entraînement FastText…")
    ft = FastText(sentences=sentences, vector_size=100, window=4, min_count=5, workers=2, sg=1, epochs=5, seed=42)
    # Embeddings film avec FastText
    embs = []
    for toks in meta["tokens"]:
        vecs = [ft.wv[w] for w in (toks or []) if w in ft.wv]
        embs.append(np.mean(vecs, axis=0) if vecs else None)
    df_ft = meta.copy()
    df_ft["has_vec"] = [e is not None for e in embs]
    V = np.vstack([e if e is not None else np.zeros(ft.vector_size) for e in embs])
    df_ft["emb"] = list(V)
    # Éval
    res = eval_w2v_classifier(df_ft, outdir=os.path.join(outdir, "fasttext"))
    return res

def tsne_plot_movies_by_genre(df_emb: pd.DataFrame, outdir: str, max_points: int = 2000):
    """t-SNE des films (embeddings) colorés par genre principal."""
    dfv = df_emb[df_emb["has_vec"]].reset_index(drop=True)
    if dfv.empty:
        print("[t-SNE films] Aucun embedding de film disponible.")
        return
    # Genre principal = premier de la liste
    def main_genres(lst):
        return lst[0] if isinstance(lst, list) and len(lst) > 0 else "other"
    dfv["main_genre"] = dfv["genres_list"].apply(main_genres)
    if len(dfv) > max_points:
        dfv = dfv.sample(max_points, random_state=42).reset_index(drop=True)
    M = np.vstack(dfv["emb"].values)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=42)
    Z = tsne.fit_transform(M)
    plt.figure(figsize=(10,8))
    genres = sorted(dfv["main_genre"].unique())
    for g in genres:
        mask = (dfv["main_genre"] == g).values
        plt.scatter(Z[mask,0], Z[mask,1], s=10, label=g, alpha=0.7)
    plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.title("Films — t-SNE coloré par genre principal")
    ensure_dir(outdir)
    fp = os.path.join(outdir, "tsne_movies_by_genre.png")
    plt.tight_layout(); plt.savefig(fp, dpi=160); plt.close()
    print("Figure t-SNE (films) sauvegardée →", os.path.abspath(fp))

def plot_grid_heatmaps(grid_csv: str, outdir: str):
    """Crée des heatmaps (window × min_count) pour chaque couple (vector_size, epochs)."""
    if not os.path.isfile(grid_csv):
        print("[grid] Aucun fichier de résultats trouvé pour heatmaps.")
        return
    df = pd.read_csv(grid_csv)
    groups = df.groupby(["vector_size", "epochs"])
    for (vs, ep), d in groups:
        pivot = d.pivot(index="window", columns="min_count", values="genre_agreement")
        plt.figure(figsize=(6,4))
        plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="genre_agreement")
        plt.title(f"Grid — vs={vs}, epochs={ep}")
        ensure_dir(outdir)
        fp = os.path.join(outdir, f"grid_heatmap_vs{vs}_ep{ep}.png")
        plt.tight_layout(); plt.savefig(fp, dpi=160); plt.close()
        print("Heatmap sauvegardée →", os.path.abspath(fp))

#
# =========================
# Export CSV + Dashboard/Report
# =========================
def write_neighbors_csv(model: Word2Vec, words: List[str], outdir: str, k: int = 10) -> str:
    """Écrit neighbors.csv avec les k voisins de chaque mot dans 'words'."""
    # Exclude top 200 most frequent tokens from neighbor candidates
    top_freq = set(list(model.wv.key_to_index.keys())[:200])
    # Skip extremely rare tokens to avoid spurious 0.999 clusters
    def _ok_token(token: str) -> bool:
        return len(token) > 2 and not token.isnumeric()
    rows = []
    for w in words:
        if w in model.wv:
            # get neighbors, excluding those in top_freq
            neighbors = []
            for nb, sc in model.wv.most_similar(w, topn=100):
                if nb in top_freq or nb == w or not _ok_token(nb):
                    continue
                neighbors.append((nb, sc))
                if len(neighbors) >= k:
                    break
            if neighbors:
                for nb, sc in neighbors:
                    rows.append({"query": w, "neighbor": nb, "similarity": float(sc)})
            else:
                rows.append({"query": w, "neighbor": "", "similarity": float("nan")})
        else:
            rows.append({"query": w, "neighbor": "", "similarity": float("nan")})
    df = pd.DataFrame(rows)
    ensure_dir(outdir)
    fp = os.path.join(outdir, "neighbors.csv")
    df.to_csv(fp, index=False)
    print("CSV voisins →", os.path.abspath(fp))
    return fp

def write_similar_movies_csv(df_emb: pd.DataFrame, title: str, outdir: str, topk: int = 10) -> Optional[str]:
    """Écrit similar_movies_<title>.csv si embeddings dispo, sinon None."""
    sims = similar_movies(df_emb, title, topk=topk)
    if not sims:
        print("[CSV films proches] Aucun résultat (titre absent ou sans vecteur).")
        return None
    df = pd.DataFrame(sims, columns=["title", "cosine"])
    ensure_dir(outdir)
    safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.strip().lower()) or "unknown_title"
    fp = os.path.join(outdir, f"similar_movies_{safe_title}.csv")
    df.to_csv(fp, index=False)
    print("CSV films proches →", os.path.abspath(fp))
    return fp

def write_eval_summary_csv(outdir: str) -> Optional[str]:
    """Fusionne eval_tfidf.json et eval_w2v.json en eval_summary.csv (si présents)."""
    rows = []
    for name, fn in [("tfidf", "eval_tfidf.json"), ("w2v", "eval_w2v.json")]:
        p = os.path.join(outdir, fn)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                rows.append({
                    "model": name,
                    "micro_f1": m.get("micro_f1", float("nan")),
                    "macro_f1": m.get("macro_f1", float("nan")),
                    "n_samples": m.get("n_samples", float("nan")),
                    "n_labels": m.get("n_labels", float("nan")),
                })
            except Exception:
                pass
    if not rows:
        print("[CSV éval] Aucune métrique JSON trouvée à fusionner.")
        return None
    df = pd.DataFrame(rows)
    fp = os.path.join(outdir, "eval_summary.csv")
    df.to_csv(fp, index=False)
    print("CSV évaluation →", os.path.abspath(fp))
    return fp

def make_dashboard(outdir: str, entropy: float, genre_acc: float,
                   neighbors_csv: Optional[str],
                   eval_csv: Optional[str],
                   tsne_words_png: str,
                   tsne_movies_png: str) -> Optional[str]:
    """
    Crée un visuel synthèse 'dashboard.png' :
      - panneau métriques (entropie, sanity check, F1 si dispo)
      - jusque 4 bar charts des voisins (un par mot de la liste --neighbors)
      - t-SNE (mots) + t-SNE (films par genre)
    """
    try:
        import matplotlib.image as mpimg
    except Exception:
        print("[dashboard] Impossible d'importer matplotlib.image.")
        return None

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # (0,0) : Métriques texte
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    txt = (
        "Synthèse Word2Vec\n"
        f"• Entropie du corpus: {entropy:.3f} bits\n"
        f"• Genre agreement (voisin partageant ≥1 genre): {genre_acc*100:.1f}%\n"
    )
    if eval_csv and os.path.isfile(eval_csv):
        try:
            df_eval = pd.read_csv(eval_csv)
            lines = ["• Éval (micro/macro F1):"]
            for _, r in df_eval.iterrows():
                lines.append(f"   - {r['model']}: {r['micro_f1']:.3f} / {r['macro_f1']:.3f}")
            txt += "\n".join(lines)
        except Exception:
            pass
    ax0.text(0.02, 0.98, txt, va="top", ha="left", fontsize=11)

    # (0,1) : Jusqu'à 4 bar charts de voisins (un par query)
    # On crée une sous-grille 2×2 à l'intérieur de la case (0,1)
    try:
        nb_spec = gs[0, 1].subgridspec(2, 2, wspace=0.35, hspace=0.55)
        queries = []
        df_nb = None
        if neighbors_csv and os.path.isfile(neighbors_csv):
            df_nb = pd.read_csv(neighbors_csv)
            if not df_nb.empty and "query" in df_nb.columns:
                # garde l'ordre d'apparition
                queries = [q for q in df_nb["query"].dropna().tolist() if isinstance(q, str)]
                # unique en conservant l'ordre
                seen = set()
                queries = [q for q in queries if not (q in seen or seen.add(q))]
                queries = queries[:4]

        if queries:
            for idx, q in enumerate(queries):
                ax = fig.add_subplot(nb_spec[idx // 2, idx % 2])
                sel = df_nb[df_nb["query"] == q].sort_values("similarity", ascending=True).tail(10)
                if sel.empty:
                    ax.text(0.5, 0.5, f"Aucune donnée pour «{q}»", ha="center", va="center")
                    ax.set_axis_off()
                else:
                    ax.barh(sel["neighbor"], sel["similarity"])
                    ax.set_xlabel("cosine")
                    ax.set_ylabel(q)
                    ax.set_title(f"Voisins de « {q} »", fontsize=9)
        else:
            ax_fallback = fig.add_subplot(nb_spec[:, :])
            ax_fallback.text(0.5, 0.5, "CSV voisins absent ou vide", ha="center", va="center")
            ax_fallback.set_axis_off()
    except Exception as e:
        # Fallback : un seul panneau si sous-grille indisponible
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title("Voisins (exemple)")
        if neighbors_csv and os.path.isfile(neighbors_csv):
            try:
                df_nb = pd.read_csv(neighbors_csv)
                if not df_nb.empty:
                    first_q = df_nb["query"].iloc[0]
                    dfx = df_nb[df_nb["query"] == first_q].sort_values("similarity", ascending=True).tail(10)
                    ax1.barh(dfx["neighbor"], dfx["similarity"])
                    ax1.set_xlabel("cosine")
                    ax1.set_ylabel(f'query="{first_q}"')
                else:
                    ax1.text(0.5, 0.5, "Aucune donnée", ha="center")
            except Exception:
                ax1.text(0.5, 0.5, "Erreur lecture CSV", ha="center")
        else:
            ax1.text(0.5, 0.5, "CSV voisins absent", ha="center")

    # (1,0) : t-SNE mots (image)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("t-SNE (mots)")
    if os.path.isfile(tsne_words_png):
        img = mpimg.imread(tsne_words_png)
        ax2.imshow(img)
        ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "tsne_words.png absent", ha="center")

    # (1,1) : t-SNE films (image)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("t-SNE (films par genre)")
    if os.path.isfile(tsne_movies_png):
        img = mpimg.imread(tsne_movies_png)
        ax3.imshow(img)
        ax3.axis("off")
    else:
        ax3.text(0.5, 0.5, "tsne_movies_by_genre.png absent", ha="center")

    fig.tight_layout()
    ensure_dir(outdir)
    fp = os.path.join(outdir, "dashboard.png")
    fig.savefig(fp, dpi=160)
    plt.close(fig)
    print("Dashboard →", os.path.abspath(fp))
    return fp

# grid search 
def grid_search(sentences, meta, outdir):
    vector_sizes = [100, 200]
    windows = [3, 5]
    min_counts = [3, 5]
    epochs_list = [5, 10]

    results = []
    print("\n Début de la grille d'entraînement et d'évaluation...")

    for vs in vector_sizes:
        for w in windows:
            for mc in min_counts:
                for ep in epochs_list:
                    print(f"Training with vector_size={vs}, window={w}, min_count={mc}, epochs={ep}...")
                    model = Word2Vec(
                        sentences=sentences,
                        vector_size=vs,
                        window=w,
                        min_count=mc,
                        workers=os.cpu_count() or 4,
                        sg=1, negative=10, sample=1e-5, epochs=ep, seed=42
                    )
                    df_emb = build_movie_embeddings(meta, model)
                    acc = neighbor_genre_agreement(df_emb)
                    results.append({
                        "vector_size": vs,
                        "window": w,
                        "min_count": mc,
                        "epochs": ep,
                        "genre_agreement": acc
                    })
                    print(f"Genre agreement: {acc*100:.2f}%")

    df_grid = pd.DataFrame(results)
    ensure_dir(outdir)
    grid_fp = os.path.join(outdir, "grid_results.csv")
    df_grid.to_csv(grid_fp, index=False)
    print(f"\nGrille sauvegardée → {os.path.abspath(grid_fp)}")

    best_row = df_grid.loc[df_grid["genre_agreement"].idxmax()]
    print("\n=== Meilleure configuration ===")
    print(f"vector_size={int(best_row.vector_size)}, window={int(best_row.window)}, min_count={int(best_row.min_count)}, epochs={int(best_row.epochs)}")
    print(f"Genre agreement: {best_row.genre_agreement*100:.2f}%")

#  main
def _main_genre(lst):
    return lst[0] if isinstance(lst, list) and len(lst) > 0 else "other"

def balanced_sample_by_genre(meta: pd.DataFrame, total: int = 2000, per_genre: int = 200) -> pd.DataFrame:
    meta = meta.copy()
    meta["__main_genre__"] = meta["genres_list"].apply(_main_genre)
    parts = []
    for _, d in meta.groupby("__main_genre__"):
        n = min(per_genre, len(d))
        if n > 0:
            parts.append(d.sample(n, random_state=42))
    if not parts:
        return meta.sample(min(total, len(meta)), random_state=42).drop(columns=["__main_genre__"], errors="ignore")
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=42)
    if len(out) > total:
        out = out.sample(total, random_state=42)
    return out.drop(columns=["__main_genre__"], errors="ignore")

def main():
    ap = argparse.ArgumentParser(description="Word2Vec sur The Movies Dataset (Gensim)")
    ap.add_argument("--data-dir", default=".", help="Dossier contenant les CSV Kaggle (movies_metadata.csv, keywords.csv)")
    ap.add_argument("--outdir", default=os.path.expanduser("~/w2v_out"), help="Dossier de sortie (par défaut: ~/w2v_out)")
    ap.add_argument("--title", default="Toy Story", help="Titre pour la recherche de films similaires")
    ap.add_argument("--neighbors", default="space,love,war,police,friendship,robot,family", help="Mots dont on affiche les voisins")
    ap.add_argument("--epochs", type=int, default=10, help="Epochs d'entraînement (override rapide)")
    ap.add_argument("--grid", action="store_true", help="Effectuer une recherche en grille sur les hyperparamètres")
    ap.add_argument("--load", type=str, default=None, help="Chemin d'un modèle Word2Vec déjà entraîné à recharger (évite réapprentissage)")
    ap.add_argument("--tsne", action="store_true", help="Forcer la génération du t-SNE des mots (figure)")
    ap.add_argument("--eval", action="store_true", help="Évaluer des classifieurs de genres (TF-IDF vs W2V).")
    ap.add_argument("--fasttext", action="store_true", help="Inclure FastText dans la comparaison (si disponible).")
    ap.add_argument("--no-light", action="store_true", help="Désactiver le mode léger (utiliser tout le corpus).")
    ap.add_argument("--preview", action="store_true", help="Ouvre automatiquement le dashboard et les figures générées (macOS: Aperçu/Finder).")
    # New CLI options for W2V
    ap.add_argument("--sg", type=int, default=1, choices=[0,1], help="0=CBOW, 1=Skip-gram (default: 1)")
    ap.add_argument("--window", type=int, default=4, help="Contexte (window size) pour Word2Vec (default: 4)")
    ap.add_argument("--min-count", type=int, default=5, help="Min count pour Word2Vec (default: 5)")
    ap.add_argument("--sample", type=float, default=1e-5, help="Subsampling pour Word2Vec (default: 1e-5)")
    ap.add_argument("--negative", type=int, default=10, help="Négatifs pour Word2Vec (default: 10)")
    ap.add_argument("--phrases", action="store_true", help="Active la détection de bigrammes/trigrammes (phrases)")
    ap.add_argument(
        "--recompute-tsne",
        action="store_true",
        help="Force la suppression et la régénération des fichiers t-SNE avant exécution"
    )
    # Added: vector size and pruning controls
    ap.add_argument("--vector-size", type=int, default=150, help="Taille des vecteurs Word2Vec (default: 150)")
    ap.add_argument("--max-df", type=float, default=0.25, help="Supprime les tokens présents dans >max_df fraction des documents (default: 0.25)")
    ap.add_argument("--min-df", type=int, default=5, help="Supprime les tokens présents dans <min_df documents (default: 5)")
    ap.add_argument("--no-prune-df", action="store_true", help="Désactive la coupe par fréquence documentaire")
    ap.add_argument("--no-tsne", action="store_true", help="Désactive la génération des t-SNE pour accélérer le run")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Suppression des anciens fichiers t-SNE si demandé
    if args.recompute_tsne:
        for fn in ["tsne_words.png", "tsne_movies_by_genre.png", "tsne_words.csv", "tsne_movies.csv"]:
            fp = os.path.join(args.outdir, fn)
            if os.path.isfile(fp):
                os.remove(fp)
                print(f"[info] Supprimé ancien fichier : {fp}")

    print(" Chargement & préparation du corpus…")
    t0 = time.time()
    sentences, meta = load_corpus(args.data_dir, phrases=args.phrases)
    # 🔧 Mode allégé pour MacBook Air (évite surcharge CPU/mémoire)
    if (not args.no_light) and (len(sentences) > 2000):
        meta = balanced_sample_by_genre(meta, total=2000, per_genre=200).reset_index(drop=True)
        sentences = meta["tokens"].tolist()
        print(f"[MODE LIGHT] Corpus équilibré par genre → {len(sentences)} films (≤2000, ≤200/genre).")
    else:
        print("[MODE FULL] Utilisation du corpus complet.")
    print(f"Corpus: {len(sentences)} documents (films) | tokens/doc (median) ≈ {np.median([len(s) for s in sentences]):.0f}")

    # Ensure key probe words are never pruned by DF rules
    keep_always = {w.strip().lower() for w in args.neighbors.split(",") if w.strip()} | {"space","love","war","police","friendship","robot","family"}
    # Prune high/low document-frequency tokens to avoid vector saturation
    if not args.no_prune_df:
        tokens_list = meta["tokens"].tolist()
        pruned, dfreq, removed = prune_by_docfreq(
            tokens_list,
            max_df_ratio=args.max_df,
            min_df=args.min_df,
            extra_stop=None,
            keep=keep_always,
        )
        meta["tokens"] = pruned
        sentences = pruned
        print(f"[prune] Tokens retirés par DF: {len(removed)} (df/N > {args.max_df} ou df < {args.min_df}).")
        # Build a lightweight IDF map for TF–IDF weighted film embeddings
        N_docs = max(1, len(pruned))
        idf_map = {t: math.log((N_docs + 1) / (c + 1)) + 1.0 for t, c in dfreq.items()}
    else:
        idf_map = None

    # Calcul de l'entropie du corpus
    entropy = compute_corpus_entropy(sentences)
    print(f"Entropie du corpus (distribution des tokens): {entropy:.3f} bits")

    if args.grid:
        grid_search(sentences, meta, args.outdir)
        plot_grid_heatmaps(os.path.join(args.outdir, "grid_results.csv"), args.outdir)
        print("\n Grille terminée.")
        return

    # Hyperparamètres (par défaut ou issus du grid/CLI)
    vector_size = args.vector_size
    window = args.window
    min_count = args.min_count
    epochs = args.epochs
    sg = args.sg
    negative = args.negative
    sample = args.sample
    model = None
    model_kv_fp = os.path.join(args.outdir, "w2v_movies.kv")
    model_full_fp = os.path.join(args.outdir, "w2v_full.model")

    # Chargement d'un modèle déjà entraîné si demandé
    if args.load:
        print(f"🔹 Chargement du modèle Word2Vec depuis {args.load} ...")
        if args.load.endswith(".kv"):
            from gensim.models import KeyedVectors
            model = Word2Vec(vector_size=vector_size)  # dummy, will set .wv
            model.wv = Word2Vec.load(args.load).wv
        else:
            model = Word2Vec.load(args.load)
        print("Modèle chargé.")
    else:
        print(" Entraînement Word2Vec…")
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=2,
            sg=sg,
            negative=negative,
            sample=sample,
            epochs=epochs,
            seed=42
        )
        model.wv.save(model_kv_fp)
        model.save(model_full_fp)
        print("Modèle sauvegardé →", os.path.abspath(model_kv_fp))
        print("Modèle complet sauvegardé →", os.path.abspath(model_full_fp))
        # Assure les normes unitaires sont bien en cache pour les similarités
        try:
            model.wv.fill_norms(force=True)
        except Exception:
            pass

    # voisins de mots
    words = [w.strip() for w in args.neighbors.split(",") if w.strip()]
    print(f"[params] vector_size={vector_size}, window={window}, min_count={min_count}, sg={sg}, negative={negative}, sample={sample}")
    print(f"[paths] outdir={os.path.abspath(args.outdir)}")
    print_neighbors(model, words, k=10)

    # embeddings film
    print(" Embeddings film (moyenne des tokens)…")
    df_emb = build_movie_embeddings(meta, model, idf=idf_map, tokens_col="tokens_raw")
    meta_out = meta.copy()
    meta_out["has_vec"] = df_emb["has_vec"]
    cols_keep = [c for c in ["id","title","genres_list","tokens","tokens_raw","has_vec"] if c in meta_out.columns]
    meta_out[cols_keep].to_csv(os.path.join(args.outdir, "movies_tokens.csv"), index=False)

    # t-SNE des films colorés par genre principal
    if not args.no_tsne:
        tsne_plot_movies_by_genre(df_emb, args.outdir)
    else:
        print("⏩ t-SNE des films désactivé (--no-tsne).")

    # Évaluation extrinsèque (TF-IDF vs W2V, + FastText optionnel)
    if args.eval:
        eval_tfidf_baseline(meta, args.outdir)
        eval_w2v_classifier(df_emb, args.outdir)
        if args.fasttext:
            train_fasttext_and_eval(sentences, meta, args.outdir)

    # similarité film↔film
    sims = similar_movies(df_emb, args.title, topk=10)
    if sims:
        print(f"\n=== Films proches de « {args.title} » ===")
        for t, sc in sims:
            print(f"- {t}  (cos={sc:.3f})")
    else:
        print(f"\n[info] Pas de vecteur pour le titre « {args.title} » — essaie un autre titre existant.")

    # mini-métrique genres
    acc = neighbor_genre_agreement(df_emb)
    print(f"\nSanity check: voisin partageant ≥1 genre → {acc*100:.1f}%")

    # t-SNE automatique (toujours activé)
    if not args.no_tsne:
        print(" t-SNE sur vocabulaire (échantillon)…")
        tsne_plot_words(model, args.outdir, topn=200)
    else:
        print("⏩ t-SNE des mots désactivé (--no-tsne).")

    # sauvegarde embeddings film (optionnel)
    # (attention: gros fichier si dataset complet)
    # On sauvegarde seulement les films avec vecteur
    df_save = df_emb[df_emb["has_vec"]][["id","title","genres_list","emb"]].copy()
    # emb -> chaîne JSON
    df_save["emb"] = df_save["emb"].apply(lambda v: json.dumps(v.tolist()))
    df_save.to_csv(os.path.join(args.outdir, "movie_embeddings.csv"), index=False)
    print("Embeddings film →", os.path.abspath(os.path.join(args.outdir, "movie_embeddings.csv")))

    #  Exports CSV utiles + dashboard synthèse 
    # 1) voisins pour la liste de mots passée via --neighbors
    neighbors_csv = write_neighbors_csv(model, words, args.outdir, k=10)
    # 2) films similaires au titre demandé (si dispo)
    sim_csv = write_similar_movies_csv(df_emb, args.title, args.outdir, topk=10)
    # 3) fusion des métriques d'évaluation en un CSV
    eval_csv = write_eval_summary_csv(args.outdir)

    # 4) dashboard image récapitulatif (métriques + t-SNE + voisins)
    tsne_words_png = os.path.join(args.outdir, "tsne_words.png")
    tsne_movies_png = os.path.join(args.outdir, "tsne_movies_by_genre.png")
    dash_fp = make_dashboard(args.outdir, entropy, acc, neighbors_csv, eval_csv, tsne_words_png, tsne_movies_png)

    # Sauvegarder les hyperparamètres et métriques principaux
    _save_params(
        args.outdir,
        params={
            "vector_size": getattr(model, "vector_size", None),
            "window": getattr(model, "window", None),
            "min_count": getattr(model, "min_count", None),
            "epochs": getattr(model, "epochs", None),
            "sg": getattr(model, "sg", None) if hasattr(model, "sg") else args.sg,
            "negative": getattr(model, "negative", None) if hasattr(model, "negative") else args.negative,
            "sample": getattr(model, "sample", None) if hasattr(model, "sample") else args.sample,
        },
        metrics={
            "entropy": entropy,
            "genre_agreement": acc,
            # Ajout de micro/macro F1 si disponibles dans eval_csv
            "micro_f1": None,
            "macro_f1": None
        }
    )

    # Ouvrir automatiquement les visuels si demandé
    if args.preview:
        for pth in [dash_fp, tsne_words_png, tsne_movies_png]:
            if pth and os.path.isfile(pth):
                open_file_in_os(pth)
        # Ouvre aussi le dossier de sortie pour que tout soit visible
        if os.path.isdir(args.outdir):
            open_file_in_os(args.outdir)

    t1 = time.time()
    # Résumé final
    print("\n=== Résumé final ===")
    print(f"Paramètres principaux du modèle : vector_size={getattr(model, 'vector_size', '?')}, window={getattr(model, 'window', '?')}, min_count={getattr(model, 'min_count', '?')}, epochs={getattr(model, 'epochs', '?')}")
    print(f"Entropie du corpus : {entropy:.3f} bits")
    print(f"Sanity check (genre agreement) : {acc*100:.1f}%")
    print(f"Temps d'exécution total : {t1-t0:.1f} sec")
    print("\n Terminé.")

if __name__ == "__main__":
    main()