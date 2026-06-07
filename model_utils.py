"""
Modell-Hilfsfunktionen fuer die App.

load_or_build() liefert (Pipeline, meta):
  1. Falls ein gespeichertes Modell vorhanden ist (rating_model.joblib), wird es
     geladen -> reine Inferenz.
  2. Sonst werden die Daten geholt (lokale CSV oder GitHub) und das Modell wird
     einmalig trainiert und gespeichert. So laeuft der Space auch ohne grosses
     hochgeladenes Binaerfile.

Die kanonische Trainings-/Vergleichslogik bleibt in src/train.py (Doku-Grundlage);
hier wird dieselbe beste Konfiguration verwendet.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

HERE = Path(__file__).resolve().parent
DATA_URL = ("https://raw.githubusercontent.com/harshitcodes/tmdb_movie_data_analysis/"
            "master/tmdb-5000-movie-dataset/tmdb_5000_movies.csv")
NUM = ["budget_log", "has_budget", "runtime", "year"]
TARGET = "vote_average"


def _locate(name: str):
    for p in (HERE / "models" / name, HERE / name):
        if p.exists():
            return p
    return None


def _read_data() -> pd.DataFrame:
    local = _locate("tmdb_5000_movies.csv") or (HERE / "data" / "tmdb_5000_movies.csv")
    src = local if Path(local).exists() else DATA_URL
    return pd.read_csv(src)


def _genre_tokens(s):
    try:
        return " ".join(g["name"].replace(" ", "_") for g in json.loads(s))
    except Exception:
        return ""


def _prepare(df: pd.DataFrame):
    df = df[df["vote_count"] >= 10].copy()
    df = df[df["overview"].notna()].copy()
    df["genres_str"] = df["genres"].apply(_genre_tokens)
    df["budget_log"] = np.log1p(df["budget"])
    df["has_budget"] = (df["budget"] > 0).astype(int)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    df["overview"] = df["overview"].fillna("")
    df = df[NUM + ["genres_str", "overview", TARGET]].dropna(subset=["year"])
    return df


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("genre", CountVectorizer(binary=True, token_pattern=r"[^\s]+"), "genres_str"),
        ("text", TfidfVectorizer(max_features=300, stop_words="english",
                                 min_df=5, ngram_range=(1, 2)), "overview"),
    ], remainder="drop")
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                  max_features=0.5, random_state=42, n_jobs=-1)
    return Pipeline([("prep", pre), ("model", model)])


def _build_meta(df: pd.DataFrame, metrics: dict) -> dict:
    genres = sorted({t.replace("_", " ")
                     for s in df["genres_str"] for t in s.split() if t})
    ex = df.assign(g=df["genres_str"].str.split()).explode("g").dropna(subset=["g"])
    ex["g"] = ex["g"].str.replace("_", " ")
    genre_avg = (ex.groupby("g")[TARGET].mean().round(2)
                 .sort_values(ascending=False).to_dict())
    return {
        "model_name": "RandomForest (mit Text)",
        "metrics": metrics,
        "num_features": NUM,
        "genres": genres,
        "target": TARGET,
        "rating_mean": round(float(df[TARGET].mean()), 3),
        "runtime_median": float(df["runtime"].median()),
        "year_median": int(df["year"].median()),
        "genre_avg_rating": genre_avg,
    }


def _train():
    df = _prepare(_read_data())
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    # Metriken aus einem Hold-out-Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = _build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {"mae": round(mean_absolute_error(y_te, pred), 4),
               "rmse": round(root_mean_squared_error(y_te, pred), 4),
               "r2": round(r2_score(y_te, pred), 4)}
    # Finales Modell auf allen Daten
    final = _build_pipeline().fit(X, y)
    meta = _build_meta(df, metrics)
    # Speichern fuer schnellere naechste Starts
    try:
        joblib.dump(final, HERE / "rating_model.joblib")
        (HERE / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    except Exception:
        pass
    return final, meta


def load_or_build():
    mp = _locate("rating_model.joblib")
    metap = _locate("meta.json")
    if mp and metap:
        return joblib.load(mp), json.loads(Path(metap).read_text())
    print("Kein Modell gefunden – trainiere einmalig beim Start ...")
    return _train()
