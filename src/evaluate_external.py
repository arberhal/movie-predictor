"""
Out-of-Sample-Evaluation auf der ZWEITEN Datenquelle (data/recent_movies.csv):
bekannte Filme 2022-2024, die NICHT im Trainingsdatensatz (TMDB 5000, bis ~2017)
enthalten sind. Vergleicht die Modellvorhersage mit der realen IMDb-Bewertung.

Zweck: zeigt, ob das Modell auf neue, unabhaengige Daten generalisiert
(-> Fehleranalyse fuer die Dokumentation).

Hinweis: IMDb- und TMDB-Bewertung sind verschiedene Plattformen (beide /10,
stark korreliert). Der Vergleich ist daher naeherungsweise zu lesen.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_utils import load_or_build

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

MODEL, META = load_or_build()
df = pd.read_csv(ROOT / "data" / "recent_movies.csv")


def to_genres_str(s):
    return " ".join(g.strip().replace(" ", "_") for g in str(s).split(","))


feat = pd.DataFrame({
    "budget_log": np.log1p(df["budget"]),
    "has_budget": (df["budget"] > 0).astype(int),
    "runtime": df["runtime"],
    "year": df["year"],
    "genres_str": df["genres"].apply(to_genres_str),
    "overview": df["overview"].fillna(""),
})

df["predicted"] = np.clip(MODEL.predict(feat), 0, 10).round(2)
df["error"] = (df["predicted"] - df["actual_rating"]).round(2)

mae = df["error"].abs().mean()
from sklearn.metrics import r2_score
r2 = r2_score(df["actual_rating"], df["predicted"])
print("=== Out-of-Sample-Test auf neuen Filmen (zweite Datenquelle) ===\n")
print(df[["title", "actual_rating", "predicted", "error"]].to_string(index=False))
print(f"\nMAE: {mae:.3f} | R2: {r2:.3f}  (Test-MAE im Training: {META['metrics']['mae']})")

# Plot: Vorhersage vs. reale Bewertung; nur die groessten Ausreisser beschriften
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df["actual_rating"], df["predicted"], color="#4C72B0", s=35, alpha=.8)
outliers = df.reindex(df["error"].abs().sort_values(ascending=False).index).head(6)
for _, r in outliers.iterrows():
    ax.annotate(r["title"][:16], (r["actual_rating"], r["predicted"]),
                fontsize=7, xytext=(4, 2), textcoords="offset points")
ax.plot([3, 9], [3, 9], "r--", lw=1, label="perfekte Vorhersage")
ax.set_xlabel("Reale IMDb-Bewertung"); ax.set_ylabel("Modell-Vorhersage")
ax.set_title(f"36 neue Filme 2022-2024\nMAE={mae:.2f}, R²={r2:.2f}")
ax.set_xlim(3, 9); ax.set_ylim(3, 9); ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "05_external_eval.png", dpi=120); plt.close(fig)

df.to_csv(ROOT / "data" / "recent_movies_results.csv", index=False)
print("\nPlot -> docs/figures/05_external_eval.png, Ergebnisse -> data/recent_movies_results.csv")
