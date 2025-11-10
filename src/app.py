import logging
from pathlib import Path
import os
import subprocess
import difflib
from typing import Tuple, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State

logging.basicConfig(level=logging.INFO, format="[Dash] %(message)s")

def ensure_fresh_tsne(outdir="w2v_out", model_path="w2v_out/w2v_movies.kv", script_path=None):
    """Vérifie si les fichiers t-SNE sont à jour par rapport au modèle, sinon les régénère.

    - outdir/model_path peuvent être des str ou des Path.
    - script_path: chemin absolu du script w2v_movies.py à exécuter. Si None, on le déduit du présent fichier.
    """
    outdir = Path(outdir)
    model = Path(model_path)
    if script_path is None:
        # /.../w2v_project/src/app.py -> /.../w2v_project/src/w2v_movies.py
        script_path = Path(__file__).resolve().parent / "w2v_movies.py"
    else:
        script_path = Path(script_path)

    tsne_files = [outdir / "tsne_words.csv", outdir / "tsne_movies.csv"]
    if not model.exists():
        print("[Avertissement] Modèle Word2Vec introuvable — impossible de régénérer t-SNE.")
        return

    need_recompute = False
    try:
        model_mtime = model.stat().st_mtime
        for f in tsne_files:
            if (not f.exists()) or (f.stat().st_mtime < model_mtime):
                need_recompute = True
                break
    except Exception as e:
        print(f"[Avertissement] Impossible de comparer les dates de fichiers: {e}")
        need_recompute = True

    if need_recompute:
        print("Recalcul du t-SNE car le modèle est plus récent...")
        try:
            subprocess.run(
                ["/opt/anaconda3/bin/python", str(script_path), "--recompute-tsne"],
                check=False
            )
        except Exception as e:
            print(f"[Erreur] Impossible de régénérer le t-SNE : {e}")
    else:
        print("Fichiers t-SNE déjà à jour.")

# =====================================================================
# Chemins par défaut
# =====================================================================
def _pick_best_w2v_out(primary: Path, fallback: Path) -> Path:
    """Choose the w2v_out folder with the freshest model (by mtime). Falls back to the one with more expected files."""
    def info(p: Path):
        if not p.exists():
            return (-1, 0)  # (mtime, score)
        kv = p / "w2v_movies.kv"
        try:
            mtime = kv.stat().st_mtime if kv.exists() else -1
        except Exception:
            mtime = -1
        wanted = {"w2v_movies.kv","movie_embeddings.csv","neighbors.csv","tsne_movies.csv","tsne_words.csv","params_metrics.json"}
        try:
            names = {x.name for x in p.iterdir() if x.is_file()}
        except Exception:
            names = set()
        score = len(wanted & names)
        return (mtime, score)

    p_mtime, p_score = info(primary)
    f_mtime, f_score = info(fallback)

    # Prefer freshest model by mtime
    if p_mtime != f_mtime:
        return primary if p_mtime > f_mtime else fallback
    # Else prefer higher score
    if p_score != f_score:
        return primary if p_score >= f_score else fallback
    # Else prefer primary if it exists, otherwise fallback
    return primary if primary.exists() else fallback

BASE_DIR = Path(__file__).resolve().parents[1]  # /Users/cakinmevlut/Desktop/w2v_project
PRIMARY_W2V_OUT = BASE_DIR / "w2v_out"
ALT_W2V_OUT = Path("/Users/cakinmevlut/w2v_out")

# Choisir le dossier le plus pertinent
if PRIMARY_W2V_OUT.exists() and any(PRIMARY_W2V_OUT.iterdir()):
    BASE_W2V_OUT = _pick_best_w2v_out(PRIMARY_W2V_OUT, ALT_W2V_OUT)
elif ALT_W2V_OUT.exists():
    BASE_W2V_OUT = ALT_W2V_OUT
else:
    BASE_W2V_OUT = PRIMARY_W2V_OUT  # par défaut

CSV_NEIGHBORS = BASE_W2V_OUT / "neighbors.csv"

# Fichiers éventuels produits par le script d'entraînement
KV_MODEL = BASE_W2V_OUT / "w2v_movies.kv"
MOVIE_EMB = BASE_W2V_OUT / "movie_embeddings.csv"

# --- Cached loader for KeyedVectors (used to recompute neighbors on the fly)
_KV_CACHE = {"path": None, "mtime": None, "kv": None}

def _load_kv_cached(model_path: Path = KV_MODEL):
    """Load gensim KeyedVectors and cache it, but invalidate the cache if the file path or mtime changed."""
    global _KV_CACHE
    model_path = Path(model_path)
    mtime = None
    try:
        mtime = model_path.stat().st_mtime
    except Exception:
        pass

    if (_KV_CACHE.get("kv") is not None
        and _KV_CACHE.get("path") == str(model_path)
        and _KV_CACHE.get("mtime") == mtime):
        return _KV_CACHE["kv"]

    # (Re)load
    kv = None
    try:
        kv = KeyedVectors.load(str(model_path), mmap='r')
    except Exception:
        try:
            kv = KeyedVectors.load_word2vec_format(str(model_path), binary=True)
        except Exception as e:
            logging.warning(f"Impossible de charger KeyedVectors depuis {model_path}: {e}")
            kv = None

    _KV_CACHE.update({"path": str(model_path), "mtime": mtime, "kv": kv})
    return kv

# =====================================================================
# Helpers de chargement (réutilisés dans les callbacks pour 'live reload')
# =====================================================================

def _normalize_neighbors_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Essaie de détecter query/neighbor/similarity et renomme proprement."""
    if df.empty:
        return pd.DataFrame(columns=["query", "neighbor", "similarity"])

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "query": ["query", "word", "mot", "token", "centre", "center", "pivot"],
        "neighbor": ["neighbor", "voisin", "nearest", "word2", "neighbor_word", "voisin_word"],
        "similarity": ["similarity", "similarité", "score", "cosine", "cosinus", "sim"]
    }

    colmap = {}
    for target, al in aliases.items():
        for a in al:
            if a in df.columns:
                colmap[target] = a
                break

    # Similarité manquante → meilleure colonne numérique dans [0,1]
    if "similarity" not in colmap:
        numcands = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numcands:
            best, bestscore = None, -1.0
            for c in numcands:
                s = pd.to_numeric(df[c], errors="coerce")
                score = s.dropna().between(0, 1).mean()
                if score > bestscore:
                    best, bestscore = c, score
            if best is not None:
                colmap["similarity"] = best

    # Query/neighbor manquants → 2 premières colonnes texte
    if "query" not in colmap or "neighbor" not in colmap:
        strcols = [c for c in df.columns if df[c].dtype == object]
        if len(strcols) >= 2:
            colmap.setdefault("query", strcols[0])
            colmap.setdefault("neighbor", strcols[1])

    if set(["query", "neighbor", "similarity"]).issubset(colmap):
        out = df.rename(columns={
            colmap["query"]: "query",
            colmap["neighbor"]: "neighbor",
            colmap["similarity"]: "similarity",
        }).copy()
        out["similarity"] = pd.to_numeric(out["similarity"], errors="coerce")
        out.dropna(subset=["query", "neighbor", "similarity"], inplace=True)
        return out
    else:
        logging.error(f"Colonnes manquantes dans neighbors.csv — détecté: {colmap}.")
        return pd.DataFrame(columns=["query", "neighbor", "similarity"])


def load_neighbors_df(csv_path: Path = CSV_NEIGHBORS) -> Tuple[pd.DataFrame, List[str], str | None]:
    """Charge neighbors.csv, normalise et retourne (df, WORDS, DEFAULT_WORD)."""
    try:
        raw = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        logging.warning(f"neighbors.csv introuvable/illisible: {e}")
        raw = pd.DataFrame()

    df = _normalize_neighbors_columns(raw)
    words = sorted(df["query"].unique()) if not df.empty else []
    default_word = words[0] if words else None
    logging.info(f"neighbors_df: colonnes={df.columns.tolist()} | lignes={len(df)}")
    return df, words, default_word


def find_tsne_files(base_dir: Path) -> List[Path]:
    """Recherche automatique des fichiers t-SNE dans un dossier."""
    return list(base_dir.glob("**/*tsne*.csv"))


def load_tsne_data(path: Path, label_col: str) -> pd.DataFrame:
    """Charge un CSV t-SNE (x,y, + colonne label devinée si absente)."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if not {"x", "y"}.issubset(df.columns):
        return pd.DataFrame()

    label_low = label_col.lower()
    if label_low not in df.columns:
        for cand in ["title", "word", "movie", "name", "label"]:
            if cand in df.columns:
                df[label_low] = df[cand].astype(str)
                break
        else:
            df[label_low] = df.index.astype(str)
    else:
        df[label_low] = df[label_low].astype(str)
    return df


def compute_tsne_words(kv_path: Path, n_words: int = 600, out_csv: Path | None = None) -> pd.DataFrame:
    """Calcule t-SNE des mots depuis un modèle KeyedVectors et sauvegarde en CSV si demandé."""
    if not kv_path.exists():
        logging.warning(f"Modèle KV introuvable: {kv_path}")
        return pd.DataFrame()
    try:
        kv = KeyedVectors.load(str(kv_path), mmap='r')
    except Exception:
        try:
            kv = KeyedVectors.load_word2vec_format(str(kv_path), binary=True)
        except Exception as e:
            logging.error(f"Impossible de charger KeyedVectors: {e}")
            return pd.DataFrame()
    words = kv.index_to_key[:n_words]
    if not words:
        return pd.DataFrame()
    X = np.vstack([kv[w] for w in words])
    ts = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(words)//10)), init="random", learning_rate="auto")
    XY = ts.fit_transform(X)
    df = pd.DataFrame({"x": XY[:, 0], "y": XY[:, 1], "word": words})
    if out_csv:
        try:
            df.to_csv(out_csv, index=False)
        except Exception as e:
            logging.warning(f"Échec sauvegarde t-SNE words: {e}")
    return df


def compute_tsne_movies(emb_csv: Path, n_movies: int = 1500, out_csv: Path | None = None) -> pd.DataFrame:
    """Calcule t-SNE des films depuis movie_embeddings.csv (colonnes numériques) et sauvegarde en CSV."""
    if not emb_csv.exists():
        logging.warning(f"Embeddings films introuvables: {emb_csv}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(emb_csv)
    except Exception as e:
        logging.error(f"Chargement embeddings films échoué: {e}")
        return pd.DataFrame()
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        logging.error("Aucune colonne numérique trouvée dans movie_embeddings.csv")
        return pd.DataFrame()
    # Limiter pour rapidité
    df = df.iloc[:n_movies].copy()
    X = df[num_cols].fillna(0).to_numpy()
    ts = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(df)//10)), init="random", learning_rate="auto")
    XY = ts.fit_transform(X)
    out = pd.DataFrame({"x": XY[:, 0], "y": XY[:, 1]})
    # Propager des colonnes utiles si présentes
    if "title" in df.columns:
        out["title"] = df["title"].astype(str).values
    if "genres" in df.columns:
        out["genres"] = df["genres"].astype(str).values
    if out_csv:
        try:
            out.to_csv(out_csv, index=False)
        except Exception as e:
            logging.warning(f"Échec sauvegarde t-SNE films: {e}")
    return out

# =====================================================================
# Chargement initial (sera aussi refait périodiquement via Interval)
# =====================================================================

ensure_fresh_tsne(BASE_W2V_OUT, KV_MODEL, script_path=BASE_DIR / "src" / "w2v_movies.py")
NEI_DF, WORDS, DEFAULT_WORD = load_neighbors_df()

tsne_candidates = find_tsne_files(BASE_W2V_OUT)
TSNE_WORDS_PATH = next((p for p in tsne_candidates if "word" in p.name.lower()), BASE_W2V_OUT / "tsne_words.csv")
TSNE_MOVIES_PATH = next((p for p in tsne_candidates if "movie" in p.name.lower()), BASE_W2V_OUT / "tsne_movies.csv")

TSNE_WORDS = load_tsne_data(TSNE_WORDS_PATH, "word")
TSNE_MOVIES = load_tsne_data(TSNE_MOVIES_PATH, "title")

TSNE_OPTIONS = [
    {"label": "Espace des mots (Word2Vec)", "value": "words"},
    {"label": "Espace des films (moyenne des tokens)", "value": "movies"},
]

# 
# Application Dash
# 

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Exploration Word2Vec"

# On crée TOUJOURS les composants (même si df vides) pour permettre le 'live reload'
tab_neighbors = dbc.Container([
    html.H2("Exploration des voisins de mots"),
    html.Hr(),
    html.Small(f"Source active : {BASE_W2V_OUT}  |  Modèle : {KV_MODEL.name}", style={"opacity": 0.6}),
    html.Label("Choisir un mot :"),
    dcc.Dropdown(
        id="word-dropdown",
        options=[{"label": w, "value": w} for w in WORDS],
        value=DEFAULT_WORD,
        placeholder="Aucun mot disponible (neighbors.csv manquant)",
        style={"width": "50%"}
    ),
    dcc.Graph(id="neighbors-graph", style={"height": "600px"}),
    html.Div(id="debug-msg", style={"fontSize": "12px", "marginTop": "6px", "opacity": 0.7}),
    # Interval pour recharger les fichiers toutes les 5s
    dcc.Interval(id="fs-watch", interval=5000, n_intervals=0)
])

tab_tsne = dbc.Container([
    html.H2("Carte sémantique t‑SNE (filtrable)"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Type d'embedding :"),
            dcc.RadioItems(
                id="tsne-mode",
                options=[
                    {"label": "Espace des mots (Word2Vec)", "value": "words"},
                    {"label": "Espace des films (moyenne des tokens)", "value": "movies"},
                ],
                value="words",
                inline=True,
                inputStyle={"marginRight": "6px"},
                style={"marginBottom": "10px"}
            ),
        ], width=12),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col([
            html.Label("Filtrer par mot / titre :"),
            dcc.Input(
                id="tsne-filter",
                type="text",
                placeholder="ex : love, robot, toy...",
                debounce=True,
                style={"width": "100%"}
            ),
        ], sm=12, md=6, lg=5),
        dbc.Col([
            html.Label("Nombre de points :"),
            dcc.Slider(
                id="tsne-n",
                min=200, max=2000, step=100, value=800,
                marks={200: "200", 800: "800", 1500: "1500", 2000: "2000"},
                tooltip={"always_visible": False}
            ),
        ], sm=12, md=6, lg=5),
        dbc.Col([
            html.Label("Affichage :"),
            dcc.Checklist(
                id="tsne-labels",
                options=[{"label": "Afficher les labels", "value": "labels"}],
                value=[],
                inline=True
            ),
        ], sm=12, md=12, lg=2),
    ], className="mb-2"),
    dcc.Graph(id="tsne-graph", style={"height": "700px"}),
    html.Div(id="tsne-empty-msg", style={"fontSize": "12px", "marginTop": "6px", "opacity": 0.7}),
])
# ======================== Nouvelle Callback t-SNE Dual ==========================

@app.callback(
    Output("tsne-words-graph", "figure"),
    Output("tsne-movies-graph", "figure"),
    Output("tsne-msg", "children"),
    Input("recompute-tsne-btn", "n_clicks"),
)
def update_tsne_dual(n_clicks):
    """Affiche deux projections t-SNE (mots et films), recalcule si bouton cliqué ou fichiers absents."""
    import time
    msg = []
    # Recherche/chargement des CSV existants
    words_path = TSNE_WORDS_PATH
    movies_path = TSNE_MOVIES_PATH
    recompute = False
    if n_clicks and n_clicks > 0:
        recompute = True
        msg.append("🔁 Recalcul demandé par l'utilisateur.")
    # Charger ou recalculer les t-SNE
    ts_words = load_tsne_data(words_path, "word") if words_path.exists() else pd.DataFrame()
    ts_movies = load_tsne_data(movies_path, "title") if movies_path.exists() else pd.DataFrame()
    # Si absent ou bouton, recalculer
    if ts_words.empty or recompute:
        ts_words = compute_tsne_words(KV_MODEL, n_words=600, out_csv=words_path)
        msg.append(f"t-SNE mots recalculé ({len(ts_words)} points).")
    else:
        msg.append(f"{len(ts_words)} mots chargés.")
    if ts_movies.empty or recompute:
        ts_movies = compute_tsne_movies(MOVIE_EMB, n_movies=1500, out_csv=movies_path)
        msg.append(f"t-SNE films recalculé ({len(ts_movies)} points).")
    else:
        msg.append(f"{len(ts_movies)} films chargés.")
    # Graph mots
    if not ts_words.empty:
        fig_words = px.scatter(
            ts_words, x="x", y="y", text="word",
            title="Projection t-SNE des mots (Word2Vec)",
            template="plotly_white"
        )
        fig_words.update_traces(marker=dict(size=6, opacity=0.7))
        fig_words.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=40))
    else:
        fig_words = px.scatter(title="Aucune donnée de mots disponible")
        fig_words.update_layout(template="plotly_white", title_x=0.5)
    # Graph films
    if not ts_movies.empty:
        color_col = "genres" if "genres" in ts_movies.columns else None
        fig_movies = px.scatter(
            ts_movies, x="x", y="y", hover_name="title",
            color=color_col,
            title="Projection t-SNE des films (moyenne des tokens)",
            template="plotly_white"
        )
        fig_movies.update_traces(marker=dict(size=6, opacity=0.7))
        fig_movies.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=40))
    else:
        fig_movies = px.scatter(title="Aucune donnée de films disponible")
        fig_movies.update_layout(template="plotly_white", title_x=0.5)
    # Message
    msg_str = " | ".join(msg)
    return fig_words, fig_movies, msg_str


# ======================== Nouvelle Tab Synthèse & Analyse ==========================
import json
from dash.dash_table import DataTable

def _load_model_params(base_dir=BASE_W2V_OUT):
    """Charge les paramètres/metrics du modèle depuis params_metrics.json (prioritaire) ou params.json.

    Le fichier peut être structuré:
      - à plat: {"vector_size":..., "window":..., "min_count":..., "epochs":..., "entropie":..., "sanity_check":...}
      - ou avec des sections: {"params": {...}, "metrics": {...}}
    """
    base_dir = Path(base_dir)
    candidates = [base_dir / "params_metrics.json", base_dir / "params.json"]
    raw = {}
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                break
            except Exception:
                pass

    # Fusionner éventuelles sections
    if isinstance(raw, dict):
        merged = {}
        if "params" in raw and isinstance(raw["params"], dict):
            merged.update(raw["params"])
        if "metrics" in raw and isinstance(raw["metrics"], dict):
            merged.update(raw["metrics"])
        # si rien de tout ça, considérer raw tel quel
        if not merged:
            merged = raw
    else:
        merged = {}

    # Normaliser les clés utiles
    def pick(*keys, default=None):
        for k in keys:
            if k in merged:
                return merged[k]
        return default

    params = {
        "vector_size": pick("vector_size", "size", default=100),
        "window": pick("window", default=5),
        "min_count": pick("min_count", default=2),
        "epochs": pick("epochs", default=5),
        "entropie": pick("entropie", "entropy", default=None),
        "sanity_check": pick("sanity_check", "genre_agreement", default=None),
    }
    return params

def _get_summary_table(params):
    """Retourne une table HTML des paramètres du modèle."""
    rows = []
    for k, v in params.items():
        rows.append(html.Tr([html.Td(str(k)), html.Td(str(v) if v is not None else "—")]))
    return html.Table(
        [html.Thead(html.Tr([html.Th("Paramètre"), html.Th("Valeur")]))] +
        [html.Tbody(rows)],
        style={"width": "70%", "marginBottom": "2em", "borderCollapse": "collapse", "fontSize": "16px"}
    )

tab_summary = dbc.Container([
    html.H2("Synthèse & Analyse du modèle Word2Vec"),
    html.Small(f"Données lues depuis : {BASE_W2V_OUT}", style={"opacity": 0.6}),
    html.Hr(),
    html.Div(id="summary-table-container"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="summary-similarity-hist", style={"height": "350px"}),
        ], width=6),
        dbc.Col([
            html.H4("Matrice de similarité (Top 20 voisins)", style={"textAlign": "center", "marginBottom": "0.5rem"}),
            dcc.Graph(id="summary-heatmap", style={"height": "420px"}),
        ], width=6),
    ], style={"marginBottom": "2em"}),
    html.Div(id="summary-interpretation", style={"fontSize": "17px", "background": "#f8f9fa", "padding": "1em", "borderRadius": "8px"}),
    dcc.Interval(id="summary-interval", interval=5000, n_intervals=0)
])

app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(tab_neighbors, label="Voisins de mots"),
        dbc.Tab(tab_tsne, label="Carte sémantique (t-SNE)"),
        dbc.Tab(tab_summary, label="Synthèse & Analyse"),
    ])
], fluid=True)
# =====================================================================
# Callbacks
# =====================================================================


# --- Synthèse & Analyse callbacks ---
@app.callback(
    Output("summary-table-container", "children"),
    Input("summary-interval", "n_intervals"),
)
def update_summary_table(_n):
    params = _load_model_params()
    return _get_summary_table(params)


@app.callback(
    Output("summary-similarity-hist", "figure"),
    Output("summary-heatmap", "figure"),
    Input("summary-interval", "n_intervals"),
)
def update_summary_graphs(_n):
    # Histogramme similarités
    df, _, _ = load_neighbors_df()
    if not df.empty and "similarity" in df.columns:
        sim_vals = pd.to_numeric(df["similarity"], errors="coerce").dropna()
        hist_fig = px.histogram(sim_vals, nbins=30, title="Distribution des similarités (cosinus)", template="plotly_white")
        hist_fig.update_layout(margin=dict(l=40, r=20, t=50, b=40), title_x=0.5)
    else:
        hist_fig = px.histogram(title="Aucune donnée de similarité disponible")
        hist_fig.update_layout(template="plotly_white", title_x=0.5)

    # --- Nouvelle heatmap robuste : (top queries) × (top neighbors)
    if df.empty:
        heatmap_fig = go.Figure()
        heatmap_fig.update_layout(
            title="Pas assez de données pour la heatmap",
            template="plotly_white",
            title_x=0.5,
            margin=dict(l=60, r=40, t=40, b=80)
        )
        return hist_fig, heatmap_fig

    tmp = df.copy()
    # Nettoyage des colonnes utiles
    tmp["similarity"] = pd.to_numeric(tmp["similarity"], errors="coerce")
    tmp.dropna(subset=["query", "neighbor", "similarity"], inplace=True)

    if tmp.empty:
        heatmap_fig = go.Figure()
        heatmap_fig.update_layout(
            title="Pas assez de données pour la heatmap",
            template="plotly_white",
            title_x=0.5,
            margin=dict(l=60, r=40, t=40, b=80)
        )
        return hist_fig, heatmap_fig

    # Sélectionner les requêtes les plus fréquentes
    top_queries = tmp["query"].value_counts().head(12).index.tolist()
    sub = tmp[tmp["query"].isin(top_queries)]

    # Choisir les voisins les plus pertinents (moyenne de similarité) sur ces requêtes
    top_neighbors = (
        sub.groupby("neighbor")["similarity"]
           .mean()
           .sort_values(ascending=False)
           .head(15)
           .index
           .tolist()
    )
    if not top_neighbors:
        # Secours : voisins les plus fréquents
        top_neighbors = sub["neighbor"].value_counts().head(15).index.tolist()

    # Construire la matrice |queries| × |neighbors|
    mat = np.full((len(top_queries), len(top_neighbors)), np.nan)
    for i, q in enumerate(top_queries):
        s = sub[sub["query"] == q].groupby("neighbor")["similarity"].max()
        for j, n in enumerate(top_neighbors):
            val = s.get(n, np.nan)
            try:
                mat[i, j] = float(val)
            except Exception:
                mat[i, j] = np.nan

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=top_neighbors,
        y=top_queries,
        colorscale="Viridis",
        colorbar=dict(title="Similarité (cosine)"),
        zmin=0.0, zmax=1.0
    ))
    heatmap_fig.update_layout(
        template="plotly_white",
        title="Matrice de similarité (top requêtes × top voisins)",
        title_x=0.5,
        margin=dict(l=120, r=40, t=50, b=120)
    )
    heatmap_fig.update_xaxes(tickangle=-45, automargin=True)
    heatmap_fig.update_yaxes(automargin=True)

    return hist_fig, heatmap_fig


@app.callback(
    Output("summary-interpretation", "children"),
    Input("summary-interval", "n_intervals"),
)
def update_interpretation(_n):
    params = _load_model_params()
    df, _, _ = load_neighbors_df()
    sim_mean, sim_std, sim_min, sim_max = None, None, None, None
    entropie = params.get("entropie", None)
    sanity = params.get("sanity_check", None)

    if not df.empty and "similarity" in df.columns:
        sim_vals = df["similarity"].dropna()
        sim_mean = float(sim_vals.mean())
        sim_std = float(sim_vals.std())
        sim_min = float(sim_vals.min())
        sim_max = float(sim_vals.max())

    lines = []
    lines.append(f"**Paramètres du modèle** : vecteur = {params.get('vector_size')}, fenêtre = {params.get('window')}, min_count = {params.get('min_count')}, epochs = {params.get('epochs')}")
    if entropie is not None:
        try:
            lines.append(f"**Entropie** du corpus : {float(entropie):.3f}")
        except Exception:
            lines.append(f"**Entropie** du corpus : {entropie}")
    if sanity is not None:
        try:
            lines.append(f"**Sanity check (genres en commun)** : {float(sanity):.1f}%")
        except Exception:
            lines.append(f"**Sanity check (genres en commun)** : {sanity}")

    if sim_mean is not None:
        lines.append(f"**Similarité moyenne** entre voisins : {sim_mean:.3f} (écart-type {sim_std:.3f})")
        lines.append(f"**Intervalle des similarités** : de {sim_min:.3f} à {sim_max:.3f}")
        if sim_mean > 0.7:
            perf = "Le modèle distingue bien les relations sémantiques : les mots voisins sont fortement similaires."
        elif sim_mean > 0.5:
            perf = "Le modèle capture des relations correctes, mais la similarité moyenne reste modérée."
        else:
            perf = "La similarité moyenne est faible : le modèle pourrait être amélioré (plus de données, plus d'epochs ?)."
        lines.append(f"**Interprétation** : {perf}")
    else:
        lines.append("Aucune statistique de similarité disponible (neighbors.csv vide ou absent).")

    return html.Ul([html.Li(html.Span(l)) for l in lines])

# =====================================================================
# Callbacks
# =====================================================================

@app.callback(
    Output("word-dropdown", "options"),
    Output("word-dropdown", "value"),
    Output("debug-msg", "children"),
    Input("fs-watch", "n_intervals"),
    State("word-dropdown", "value"),
)
def refresh_dropdown(_tick, current_value):
    """Recharge neighbors.csv périodiquement et met à jour les options."""
    df, words, default_word = load_neighbors_df()
    msg = f"{len(df)} lignes chargées depuis {CSV_NEIGHBORS}."
    if not words:
        return [], None, f"{msg} (aucun mot disponible)."
    # Conserver la sélection si encore valide, sinon valeur par défaut
    new_value = current_value if current_value in words else default_word
    return ([{"label": w, "value": w} for w in words], new_value, msg)


@app.callback(
    Output("neighbors-graph", "figure"),
    Input("word-dropdown", "value"),
    Input("fs-watch", "n_intervals"),
    prevent_initial_call=False
)
def update_neighbors(word, _tick):
    """Bar chart horizontal des voisins (top-15). Recharge df à chaque tick."""
    try:
        df, words, _ = load_neighbors_df()
        if (df.empty and not word) or (not words and not word):
            fig = px.bar(title="Aucune donnée de voisins disponible.")
            fig.update_layout(template="plotly_white")
            return fig

        # --- 1) Read neighbors from CSV if available for the selected word
        sub = pd.DataFrame(columns=["neighbor", "similarity"]) if df.empty else df.loc[df["query"] == word].copy()

        # If the exact word is missing, propose the closest existing one
        subtitle = ""
        if sub.empty and words:
            close = difflib.get_close_matches(word or "", words, n=1)
            if close:
                sub = df.loc[df["query"] == close[0]].copy()
                subtitle = f" (mot introuvable, proche: « {close[0]} »)"

        # --- 2) Heuristic: if similarities look saturated (all ≈1.0 or variance ~0),
        # fall back to recomputing neighbors directly from the KV model to get fresh scores
        recomputed_from_model = False
        need_model = sub.empty
        if not sub.empty:
            sim = pd.to_numeric(sub["similarity"], errors="coerce")
            if (sim.nunique(dropna=True) <= 3) or (float(sim.std(skipna=True) or 0.0) < 1e-6) or (float(sim.mean(skipna=True) or 0.0) > 0.98):
                need_model = True
        if need_model:
            kv = _load_kv_cached(KV_MODEL)
            if kv is not None and (word in kv.key_to_index or (words and difflib.get_close_matches(word or "", kv.key_to_index.keys(), n=1))):
                w = word
                if w not in kv.key_to_index:
                    # try a close match inside the kv vocab
                    close = difflib.get_close_matches(word or "", list(kv.key_to_index.keys()), n=1)
                    if close:
                        w = close[0]
                        subtitle = f" (mot introuvable dans le modèle, proche: « {w} »)"
                try:
                    pairs = kv.most_similar(w, topn=50)
                    # light denoising of ultra-generic tokens for readability
                    EXCLUDE = {"film","movie","one","two","time","way","also","first","second","third","n't"}
                    sub = pd.DataFrame(pairs, columns=["neighbor","similarity"]).query("neighbor not in @EXCLUDE")
                    recomputed_from_model = True
                except KeyError:
                    pass

        if sub.empty:
            fig = px.bar(title=f"Aucun voisin pour « {word} »")
            fig.update_layout(template="plotly_white")
            return fig

        # --- 3) Prepare bar chart (top‑15, highest cosine); zoom x‑axis to show contrast
        sub = sub.sort_values("similarity", ascending=False)
        sub = sub.groupby("neighbor", as_index=False)["similarity"].max().sort_values("similarity", ascending=False).head(15)
        sub = sub.sort_values(by="similarity").reset_index(drop=True)

        note = " (recalculés depuis le modèle)" if recomputed_from_model else subtitle
        fig = px.bar(
            sub,
            x="similarity",
            y="neighbor",
            orientation="h",
            title=f"Voisins les plus proches de « {word} »{note}",
            text="similarity",
        )
        # show more precision so 0.991/0.997 don't all look like 0.999
        fig.update_traces(
            texttemplate="%{text:.4f}",
            hovertemplate="%{y} — cos=%{x:.5f}<extra></extra>"
        )
        # adaptive zoom to reveal contrast
        smin = float(sub["similarity"].min())
        smax = float(sub["similarity"].max())
        pad = max(0.005, (smax - smin) * 0.15)
        xmin = max(0.3, smin - pad)
        xmax = min(1.0, smax + pad/2)
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="cosine",
            xaxis=dict(range=[xmin, xmax]),
            template="plotly_white",
            title_x=0.5,
            margin=dict(l=90, r=30, t=60, b=40),
        )
        return fig

    except Exception as e:
        import traceback as _tb
        logging.error(f"callback error: {e}")
        _tb.print_exc()
        fig = px.bar(title=f"Erreur: {e}")
        fig.update_layout(template="plotly_white")
        return fig


@app.callback(
    Output("tsne-graph", "figure"),
    Output("tsne-empty-msg", "children"),
    Input("tsne-mode", "value"),
    Input("tsne-filter", "value"),
    Input("tsne-n", "value"),
    Input("tsne-labels", "value"),
    Input("fs-watch", "n_intervals"),
    prevent_initial_call=False
)
def update_tsne_graph(mode, filter_text, n_points, labels_toggle, _tick):
    """Unique scatter t‑SNE (mots OU films) piloté par des filtres."""
    try:
        def labels_on():
            return isinstance(labels_toggle, (list, tuple, set)) and ("labels" in labels_toggle)

        candidates = find_tsne_files(BASE_W2V_OUT)
        words_path = next((p for p in candidates if "word" in p.name.lower()), BASE_W2V_OUT / "tsne_words.csv")
        movies_path = next((p for p in candidates if "movie" in p.name.lower()), BASE_W2V_OUT / "tsne_movies.csv")

        ts_words = load_tsne_data(words_path, "word")
        ts_movies = load_tsne_data(movies_path, "title")

        if mode == "words" and (ts_words.empty or not words_path.exists()):
            logging.info("🔁 Recalcul du t‑SNE des mots (Word2Vec)")
            ts_words = compute_tsne_words(KV_MODEL, n_words=max(600, int(n_points or 800)), out_csv=words_path)
        if mode == "movies" and (ts_movies.empty or not movies_path.exists()):
            logging.info("🔁 Recalcul du t‑SNE des films (moyenne des tokens)")
            ts_movies = compute_tsne_movies(MOVIE_EMB, n_movies=max(1000, int(n_points or 800)), out_csv=movies_path)

        if mode == "words":
            df = ts_words.copy()
            if not df.empty and n_points:
                n = max(50, int(n_points))
                if len(df) > n:
                    df = df.sample(n, random_state=42)
            if not df.empty and filter_text:
                df = df[df["word"].str.contains(str(filter_text), case=False, na=False)]
            if df.empty:
                fig = px.scatter(title="Aucune donnée de mots disponible")
                fig.update_layout(template="plotly_white", title_x=0.5)
                return fig, "⚠️ Aucun point à afficher pour les mots."
            if labels_on():
                # Afficher un sous-échantillon de labels pour éviter la surcharge
                df_label = df.sample(max(1, len(df)//10), random_state=42) if len(df) > 50 else df
                fig = px.scatter(df, x="x", y="y", hover_name="word",
                                 title="Projection t‑SNE des mots (Word2Vec)", template="plotly_white")
                fig.add_trace(go.Scatter(
                    x=df_label["x"], y=df_label["y"], text=df_label["word"],
                    mode="text", textposition="top center", textfont=dict(size=10, color="darkblue"),
                    showlegend=False
                ))
                fig.update_traces(marker=dict(size=6, opacity=0.75))
            else:
                fig = px.scatter(df, x="x", y="y", hover_name="word",
                                 title="Projection t‑SNE des mots (Word2Vec)", template="plotly_white")
                fig.update_traces(marker=dict(size=6, opacity=0.75))
            fig.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=40))
            return fig, ""

        else:
            df = ts_movies.copy()
            if not df.empty and n_points:
                n = max(100, int(n_points))
                if len(df) > n:
                    df = df.sample(n, random_state=42)
            if not df.empty and filter_text and "title" in df.columns:
                df = df[df["title"].str.contains(str(filter_text), case=False, na=False)]
            if df.empty:
                fig = px.scatter(title="Aucune donnée de films disponible")
                fig.update_layout(template="plotly_white", title_x=0.5)
                return fig, "⚠️ Aucun point à afficher pour les films."
            color_col = "genres" if "genres" in df.columns else None
            if labels_on() and "title" in df.columns:
                # Sous-échantillonnage de labels pour lisibilité
                df_label = df.sample(max(1, len(df)//10), random_state=42) if len(df) > 100 else df
                fig = px.scatter(df, x="x", y="y", color=color_col,
                                 hover_name="title",
                                 title="Projection t‑SNE des films (moyenne des tokens)", template="plotly_white")
                fig.add_trace(go.Scatter(
                    x=df_label["x"], y=df_label["y"], text=df_label["title"],
                    mode="text", textposition="top center", textfont=dict(size=9, color="black"),
                    showlegend=False
                ))
                fig.update_traces(marker=dict(size=6, opacity=0.75))
            else:
                fig = px.scatter(df, x="x", y="y", color=color_col,
                                 hover_name=("title" if "title" in df.columns else None),
                                 title="Projection t‑SNE des films (moyenne des tokens)", template="plotly_white")
                fig.update_traces(marker=dict(size=6, opacity=0.75))
            fig.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=40))
            return fig, ""
    except Exception as e:
        logging.error(f"Erreur t‑SNE: {e}")
        fig = px.scatter(title=f"Erreur: {e}")
        fig.update_layout(template="plotly_white")
        return fig, f"Erreur: {e}"

# =====================================================================
# Lancement
# =====================================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)