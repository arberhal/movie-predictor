---
title: Film-Erfolgsprognose
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
---

# 🎬 Film-Erfolgsprognose · ML + NLP

KI-Anwendung, die für einen Film die erwartete **Zuschauerbewertung** vorhersagt
und das Ergebnis in natürlicher Sprache erklärt.

## Bereiche & Integration

- **Machine Learning (numerisch):** Regressionsmodell sagt aus Eckdaten
  (Budget, Laufzeit, Jahr, Genres) die Bewertung voraus.
- **NLP:** Der Handlungstext (`overview`) wird per TF-IDF in Merkmale umgewandelt,
  die direkt ins ML-Modell einfließen. Ein LLM (OpenAI) erklärt anschließend die
  Vorhersage und schlägt eine Tagline vor.
- **Verzahnung:** gemeinsame Daten (derselbe Film), aus Text **abgeleitete
  Merkmale** (TF-IDF → Modell) und **Modellausgaben** (Vorhersage + Faktoren → LLM).
  Der Modellvergleich belegt: Die Text-Features verbessern die Vorhersage messbar
  (R² 0,325 → 0,341).

## Datensatz

TMDB 5000 Movies (The Movie Database), ~4'800 Filme mit Budget, Einspielergebnis,
Laufzeit, Genres, Handlung u. a. Strukturierte **und** textuelle Merkmale.

## Projektstruktur

```
movie-success/
├── data/tmdb_5000_movies.csv     # Rohdaten
├── src/eda.py                    # Explorative Analyse
├── src/train.py                  # Training + Modellvergleich -> models/
├── models/rating_model.joblib    # trainierte Pipeline (Inferenz)
├── models/meta.json              # Metriken + Referenzwerte
├── docs/figures/                 # EDA- & Eval-Plots
├── nlp.py                        # LLM-Erklärung + Tagline
├── app.py                        # Gradio-App (Inferenz)
└── requirements.txt
```

## Lokal ausführen

```bash
pip install -r requirements.txt
python src/eda.py          # Explorative Analyse (optional)
python src/train.py        # Modell trainieren -> models/
python app.py              # App lokal unter http://localhost:7860
```

## OpenAI-Key hinterlegen

Die App liest den Schlüssel aus der Umgebungsvariable **`OPENAI_API_KEY`**.
Ohne Schlüssel läuft die App trotzdem (regelbasierter Fallback statt LLM-Text).

- **Lokal:** vor dem Start setzen
  ```bash
  export OPENAI_API_KEY="sk-..."   # Windows: setx OPENAI_API_KEY "sk-..."
  python app.py
  ```
- **Hugging Face Spaces:** im Space unter **Settings → Variables and secrets →
  New secret** anlegen: Name `OPENAI_API_KEY`, Value dein Schlüssel.
  Niemals den Key in den Code oder ins Repo schreiben.

## Deployment (Hugging Face Spaces)

1. Neuen Space anlegen (SDK: **Gradio**).
2. Dateien hochladen / pushen (`app.py`, `nlp.py`, `requirements.txt`, `models/`).
3. Secret `OPENAI_API_KEY` setzen (siehe oben).
4. Der Space baut automatisch und stellt die öffentliche URL bereit.
