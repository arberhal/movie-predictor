"""
Explorative Datenanalyse (EDA) fuer den TMDB-5000-Datensatz.
Ziel der Anwendung: Vorhersage der Zuschauerbewertung (vote_average).
Erzeugt Kennzahlen (Konsole) und Plots (docs/figures/).
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(ROOT / "data" / "tmdb_5000_movies.csv")
print("Shape:", df.shape)

# --- 1. Datenqualitaet -------------------------------------------------
print("\n-- Datenqualitaet --")
print("budget == 0:", int((df['budget'] == 0).sum()), "Filme (fehlend)")
print("revenue == 0:", int((df['revenue'] == 0).sum()), "Filme (fehlend)")
print("runtime fehlt:", int(df['runtime'].isna().sum()))
print("overview fehlt:", int(df['overview'].isna().sum()))
print("vote_count < 10:", int((df['vote_count'] < 10).sum()),
      "Filme -> unzuverlaessige Bewertung")

# --- 2. Zielvariable: Bewertung (nur Filme mit genug Stimmen) ----------
rel = df[df['vote_count'] >= 10].copy()
print(f"\nNach Filter (vote_count >= 10): {len(rel)} Filme")
print("vote_average: mean %.2f, std %.2f" %
      (rel['vote_average'].mean(), rel['vote_average'].std()))

fig, ax = plt.subplots(figsize=(5, 3.2))
sns.histplot(rel['vote_average'], bins=30, ax=ax, color="#4C72B0")
ax.set_title("Zielverteilung: Zuschauerbewertung")
ax.set_xlabel("vote_average")
fig.tight_layout(); fig.savefig(FIG / "01_rating_distribution.png", dpi=120); plt.close(fig)

# --- 3. Genres parsen --------------------------------------------------
def genre_names(s):
    try:
        return [g["name"] for g in json.loads(s)]
    except Exception:
        return []

rel["genre_list"] = rel["genres"].apply(genre_names)
all_genres = rel.explode("genre_list")
top = all_genres["genre_list"].value_counts().head(12)
print("\n-- Haeufigste Genres --")
print(top.to_string())

# Durchschnittsbewertung je Genre
gmean = all_genres.groupby("genre_list")["vote_average"].mean().loc[top.index]
gmean = gmean.sort_values()
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.barh(gmean.index, gmean.values, color="#55A868")
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
ax.set_title("Durchschnittliche Bewertung je Genre")
ax.set_xlabel("Durchschnittliche Bewertung (vote_average)")
ax.set_ylabel("Genre")
ax.set_xlim(5.0, 6.8)   # gezoomt, damit die Unterschiede sichtbar sind
ax.axvline(gmean.mean(), color="grey", ls="--", lw=1)
fig.tight_layout(); fig.savefig(FIG / "02_rating_by_genre.png", dpi=120); plt.close(fig)

# --- 4. Korrelationen numerischer Merkmale mit der Bewertung ----------
rel["budget_log"] = np.log1p(rel["budget"])
rel["year"] = pd.to_datetime(rel["release_date"], errors="coerce").dt.year
num = rel[["budget_log", "runtime", "year", "vote_average"]].corr()["vote_average"].drop("vote_average")
print("\n-- Korrelation (nur Pre-Release-Merkmale) mit Bewertung --")
print(num.round(3).sort_values().to_string())

# Laufzeit vs Bewertung
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.scatter(rel["runtime"], rel["vote_average"], s=6, alpha=.25, color="#C44E52")
ax.set_title("Laufzeit vs. Bewertung"); ax.set_xlabel("Minuten"); ax.set_ylabel("vote_average")
ax.set_xlim(40, 220)
fig.tight_layout(); fig.savefig(FIG / "03_runtime_vs_rating.png", dpi=120); plt.close(fig)

print("\nEDA fertig. Plots in docs/figures/")
