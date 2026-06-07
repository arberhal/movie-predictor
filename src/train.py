"""
Training & Modellvergleich fuer die Film-Bewertungsprognose.

- Zielvariable: vote_average (Regression)
- Pre-Release-Merkmale (kein Leakage): Budget, Laufzeit, Jahr, Genres, Handlungstext
- Vergleich: Ridge vs. RandomForest, jeweils MIT und OHNE Text-Features
  -> zeigt, ob der NLP-Teil (TF-IDF auf der Handlung) die Vorhersage verbessert.
- Speichert die beste Pipeline nach models/ (Trennung Training/Inferenz).
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 1. Daten laden & bereinigen
# ----------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "tmdb_5000_movies.csv")
df = df[df["vote_count"] >= 10].copy()          # zuverlaessige Bewertungen
df = df[df["overview"].notna()].copy()           # Text muss vorhanden sein

def genre_tokens(s):
    try:
        return " ".join(g["name"].replace(" ", "_") for g in json.loads(s))
    except Exception:
        return ""

df["genres_str"] = df["genres"].apply(genre_tokens)
df["budget_log"] = np.log1p(df["budget"])
df["has_budget"] = (df["budget"] > 0).astype(int)
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
df["runtime"] = df["runtime"].fillna(df["runtime"].median())
df["overview"] = df["overview"].fillna("")

NUM = ["budget_log", "has_budget", "runtime", "year"]
TARGET = "vote_average"
df = df[NUM + ["genres_str", "overview", TARGET]].dropna(subset=["year"])
print(f"Trainingsdaten: {len(df)} Filme")

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------
# 2. Feature-Transformatoren
# ----------------------------------------------------------------------
def make_preprocessor(use_text: bool):
    transformers = [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("genre", CountVectorizer(binary=True, token_pattern=r"[^\s]+"), "genres_str"),
    ]
    if use_text:
        transformers.append(
            ("text", TfidfVectorizer(max_features=300, stop_words="english",
                                     min_df=5, ngram_range=(1, 2)), "overview"))
    return ColumnTransformer(transformers, remainder="drop")

def evaluate(name, model, use_text):
    pipe = Pipeline([("prep", make_preprocessor(use_text)), ("model", model)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = root_mean_squared_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print(f"{name:38s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "pipe": pipe,
            "pred": pred, "use_text": use_text}

# ----------------------------------------------------------------------
# 3. Modellvergleich
# ----------------------------------------------------------------------
print("\n=== Modellvergleich (Testset, n=%d) ===" % len(X_te))
base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
bpred = base.predict(X_te)
print(f"{'Baseline (Mittelwert)':38s} MAE={mean_absolute_error(y_te,bpred):.3f}  "
      f"RMSE={root_mean_squared_error(y_te,bpred):.3f}  R2={r2_score(y_te,bpred):.3f}")

results = []
results.append(evaluate("Ridge (ohne Text)", Ridge(alpha=1.0), False))
results.append(evaluate("Ridge (mit Text)",  Ridge(alpha=1.0), True))
results.append(evaluate("RandomForest (ohne Text)",
                        RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=0.5, random_state=42, n_jobs=-1), False))
results.append(evaluate("RandomForest (mit Text)",
                        RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=0.5, random_state=42, n_jobs=-1), True))

best = min(results, key=lambda r: r["mae"])
print(f"\nBestes Modell: {best['name']} (MAE={best['mae']:.3f})")

# ----------------------------------------------------------------------
# 4. Bestes Modell speichern (fuer die Inferenz in app.py)
# ----------------------------------------------------------------------
# Die oben berichteten Metriken stammen aus dem 80/20-Hold-out-Split.
# Fuer das DEPLOYTE Modell fitten wir die beste Konfiguration auf ALLEN Daten
# neu (konsistent mit model_utils.py).
best["pipe"].fit(X, y)
joblib.dump(best["pipe"], MODELS / "rating_model.joblib")

# Genre-Liste + Referenzstatistiken fuer das App-UI und die Erklaerungen
genre_vocab = sorted({t.replace("_", " ")
                      for s in df["genres_str"] for t in s.split() if t})

# Durchschnittsbewertung je Genre (Referenz fuer die LLM-Erklaerung)
exploded = (df.assign(g=df["genres_str"].str.split())
              .explode("g").dropna(subset=["g"]))
exploded["g"] = exploded["g"].str.replace("_", " ")
genre_avg = (exploded.groupby("g")[TARGET].mean().round(2)
             .sort_values(ascending=False).to_dict())

meta = {
    "model_name": best["name"],
    "metrics": {k: round(best[k], 4) for k in ["mae", "rmse", "r2"]},
    "num_features": NUM,
    "genres": genre_vocab,
    "target": TARGET,
    "n_train": len(X_tr), "n_test": len(X_te),
    "rating_mean": round(float(y.mean()), 3),
    "runtime_median": float(df["runtime"].median()),
    "year_median": int(df["year"].median()),
    "genre_avg_rating": genre_avg,
}
(MODELS / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print("Gespeichert: models/rating_model.joblib + meta.json")

# ----------------------------------------------------------------------
# 5. Eval-Plot: Vorhersage vs. Wirklichkeit
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(y_te, best["pred"], s=8, alpha=.3, color="#4C72B0")
ax.plot([2, 9], [2, 9], "r--", lw=1)
ax.set_xlabel("Tatsaechliche Bewertung"); ax.set_ylabel("Vorhergesagte Bewertung")
ax.set_title(f"{best['name']}\nMAE={best['mae']:.2f}, R2={best['r2']:.2f}")
fig.tight_layout(); fig.savefig(FIG / "04_pred_vs_actual.png", dpi=120); plt.close(fig)
print("Eval-Plot gespeichert.")
