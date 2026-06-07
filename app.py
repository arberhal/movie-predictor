"""
Film-Erfolgsprognose – Gradio-App (Inferenz) fuer Hugging Face Spaces.

Verbindet ML und NLP:
  Eingabe (Eckdaten + Handlung)
    -> ML-Modell sagt die Zuschauerbewertung voraus (nutzt auch Text-Features)
    -> LLM erklaert das Ergebnis und schlaegt eine Tagline vor.

Tab 2 nutzt die ZWEITE Datenquelle (recent_movies.csv): bekannte Filme
2022-2024, die nicht im Training waren -> Vergleich Vorhersage vs. Wirklichkeit.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import gradio as gr

import nlp
from model_utils import load_or_build

HERE = Path(__file__).resolve().parent
MODEL, META = load_or_build()


def _load_recent():
    for p in (HERE / "data" / "recent_movies.csv", HERE / "recent_movies.csv"):
        if p.exists():
            return pd.read_csv(p)
    return None


RECENT = _load_recent()


def featurize(budget, runtime, year, genres, overview) -> pd.DataFrame:
    """Baut genau die Spalten, die die gespeicherte Pipeline erwartet."""
    genres_str = " ".join(g.replace(" ", "_") for g in (genres or []))
    return pd.DataFrame([{
        "budget_log": np.log1p(budget or 0),
        "has_budget": 1 if (budget and budget > 0) else 0,
        "runtime": runtime,
        "year": year,
        "genres_str": genres_str,
        "overview": overview or "",
    }])


def predict(title, budget, runtime, year, genres, overview):
    if not overview or len(overview.strip()) < 15:
        return "—", "Bitte gib eine kurze Handlungsbeschreibung ein (mind. 15 Zeichen).", ""
    row = featurize(budget, runtime, year, genres, overview)
    rating = max(0.0, min(10.0, float(MODEL.predict(row)[0])))
    movie = {"title": title, "budget": budget, "runtime": runtime,
             "year": year, "genres": genres or [], "overview": overview}
    explanation = nlp.explain(movie, rating, META, variant="detailed")
    tag = nlp.tagline(movie)
    score_md = f"## ⭐ {rating:.1f} / 10\n_(Datensatz-Ø {META['rating_mean']})_"
    tag_md = f'### Vorgeschlagene Tagline\n*„{tag}"*' if tag else ""
    return score_md, explanation, tag_md


def compare_real(title):
    """Vergleicht die Modell-Vorhersage mit der echten Bewertung (2. Datenquelle)."""
    if RECENT is None:
        return "Keine Vergleichsdaten gefunden (recent_movies.csv fehlt im Space)."
    sub = RECENT[RECENT["title"] == title]
    if sub.empty:
        return "Bitte einen Film auswaehlen."
    r = sub.iloc[0]
    genres = [g.strip() for g in str(r["genres"]).split(",")]
    row = featurize(r["budget"], r["runtime"], r["year"], genres, r["overview"])
    pred = float(np.clip(MODEL.predict(row)[0], 0, 10))
    actual = float(r["actual_rating"])
    diff = pred - actual
    return (f"### {r['title']} ({int(r['year'])})\n"
            f"- **Modell-Vorhersage:** {pred:.1f} / 10\n"
            f"- **Echte IMDb-Bewertung:** {actual:.1f} / 10\n"
            f"- **Abweichung:** {diff:+.1f}\n\n"
            f"_Quelle: zweite Datenquelle (Filme 2022–2024, nicht im Training enthalten)._")


with gr.Blocks(title="Film-Erfolgsprognose") as demo:
    gr.Markdown(
        "# 🎬 Film-Erfolgsprognose\n"
        "Schaetzt die erwartete **Zuschauerbewertung** aus Eckdaten + Handlung. "
        "Das ML-Modell nutzt Zahlen *und* aus dem Handlungstext abgeleitete "
        "Merkmale; ein LLM erklaert das Ergebnis.\n\n"
        f"*Modell: {META['model_name']} · Test-MAE {META['metrics']['mae']} · "
        f"R² {META['metrics']['r2']}*"
    )

    with gr.Tabs():
        with gr.Tab("Eigenen Film bewerten"):
            with gr.Row():
                with gr.Column():
                    title = gr.Textbox(label="Titel (optional)")
                    budget = gr.Number(label="Budget (USD)", value=20_000_000)
                    runtime = gr.Slider(40, 240, value=110, step=1, label="Laufzeit (Min)")
                    year = gr.Number(label="Erscheinungsjahr", value=2024, precision=0)
                    genres = gr.CheckboxGroup(choices=META["genres"], label="Genres")
                    overview = gr.Textbox(lines=5, label="Handlung / Plot",
                                          placeholder="Worum geht es im Film?")
                    btn = gr.Button("Bewertung vorhersagen", variant="primary")
                with gr.Column():
                    out_score = gr.Markdown()
                    out_expl = gr.Markdown()
                    out_tag = gr.Markdown()
            btn.click(predict, [title, budget, runtime, year, genres, overview],
                      [out_score, out_expl, out_tag])
            gr.Examples(
                examples=[
                    ["Interstellar", 165_000_000, 169, 2014,
                     ["Adventure", "Drama", "Science Fiction"],
                     "A team of explorers travel through a wormhole in space in an "
                     "attempt to ensure humanity's survival."],
                    ["A Silly Comedy", 5_000_000, 88, 2023, ["Comedy"],
                     "Two clumsy friends accidentally start a food fight that escalates "
                     "into city-wide chaos."],
                ],
                inputs=[title, budget, runtime, year, genres, overview],
            )

        with gr.Tab("Test an echten Filmen"):
            gr.Markdown(
                "Prueft das Modell an **echten Filmen (2022–2024)** aus der zweiten "
                "Datenquelle, die es nie im Training gesehen hat. Vergleicht die "
                "Vorhersage mit der realen Bewertung."
            )
            choices = list(RECENT["title"]) if RECENT is not None else []
            film_dd = gr.Dropdown(choices=choices, label="Bekannten Film waehlen",
                                  value=(choices[0] if choices else None))
            cmp_btn = gr.Button("Vergleichen", variant="primary")
            cmp_out = gr.Markdown()
            cmp_btn.click(compare_real, film_dd, cmp_out)

if __name__ == "__main__":
    demo.launch()
