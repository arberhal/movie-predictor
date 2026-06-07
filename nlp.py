"""
NLP-Komponente: wandelt die ML-Ausgabe (vorhergesagte Bewertung) plus die
wichtigsten Einflussfaktoren in eine verstaendliche Erklaerung um und schlaegt
eine Tagline vor.

- Nutzt die OpenAI-API. Der Schluessel wird aus der Umgebungsvariable
  OPENAI_API_KEY gelesen (in Hugging Face Spaces als "Secret" hinterlegen).
- Ohne Schluessel laeuft die App trotzdem: dann greift ein regelbasierter
  Fallback, damit die Vorhersage immer sichtbar bleibt.
- Zwei Prompt-Varianten ("concise"/"detailed") fuer den geforderten
  Prompt-Vergleich im NLP-Teil.
"""
import os

LLM_MODEL = "gpt-4o-mini"


def build_drivers(movie: dict, predicted: float, meta: dict) -> list[str]:
    """Erzeugt nachvollziehbare Fakten aus Modell-Input + Referenzwerten,
    die der LLM als Begruendung verwenden kann (verankert die Erklaerung)."""
    drivers = []
    rt_med = meta.get("runtime_median", 105)
    if movie["runtime"] >= rt_med + 15:
        drivers.append(f"ueberdurchschnittliche Laufzeit ({movie['runtime']:.0f} Min, "
                       f"Median {rt_med:.0f}) – laengere Filme werden tendenziell besser bewertet")
    elif movie["runtime"] <= rt_med - 15:
        drivers.append(f"unterdurchschnittliche Laufzeit ({movie['runtime']:.0f} Min)")

    if movie["year"] and movie["year"] < meta.get("year_median", 2008):
        drivers.append("aelterer Film – aeltere Titel sind im Datensatz im Schnitt besser bewertet")

    ga = meta.get("genre_avg_rating", {})
    for g in movie["genres"]:
        if g in ga:
            tag = "hoch bewertetes" if ga[g] >= meta["rating_mean"] else "eher niedrig bewertetes"
            drivers.append(f"Genre {g}: {tag} Genre (Ø {ga[g]})")

    if movie["budget"] and movie["budget"] > 0:
        drivers.append(f"angegebenes Budget {movie['budget']:,.0f} USD")
    else:
        drivers.append("kein Budget angegeben")
    return drivers


def _fallback(predicted: float, drivers: list[str]) -> str:
    pts = "\n".join(f"- {d}" for d in drivers)
    return (f"**Vorhergesagte Bewertung: {predicted:.1f}/10**\n\n"
            f"Wichtigste Faktoren laut Modell:\n{pts}\n\n"
            f"_(Hinweis: Fuer eine ausformulierte KI-Erklaerung den OPENAI_API_KEY "
            f"als Secret hinterlegen.)_")


def explain(movie: dict, predicted: float, meta: dict, variant: str = "concise") -> str:
    """Erzeugt eine natuerlichsprachliche Erklaerung der Vorhersage."""
    drivers = build_drivers(movie, predicted, meta)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback(predicted, drivers)

    facts = "; ".join(drivers)
    genres = ", ".join(movie["genres"]) or "keine Angabe"

    if variant == "detailed":
        instruction = ("Schreibe 3-4 Saetze. Erklaere die vorhergesagte Bewertung, "
                       "gehe auf die einzelnen Faktoren ein und nenne eine moegliche "
                       "Einschraenkung der Vorhersage.")
    else:
        instruction = "Schreibe 2 knappe, klare Saetze, die die Vorhersage begruenden."

    prompt = (
        f"Ein Modell sagt fuer einen Film die Zuschauerbewertung {predicted:.1f}/10 voraus "
        f"(Datensatz-Durchschnitt {meta['rating_mean']}).\n"
        f"Titel: {movie.get('title') or 'unbenannt'}\n"
        f"Genres: {genres}\n"
        f"Handlung: {movie.get('overview', '')[:600]}\n"
        f"Vom Modell genutzte Faktoren: {facts}.\n\n"
        f"{instruction} Beziehe dich nur auf die genannten Faktoren, erfinde nichts dazu."
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Du bist ein Film-Analyst und erklaerst "
                 "Vorhersagen sachlich und verstaendlich auf Deutsch."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return _fallback(predicted, drivers) + f"\n\n_(LLM-Fehler: {e})_"


def tagline(movie: dict) -> str:
    """Schlaegt eine packende Tagline zur Handlung vor."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not movie.get("overview"):
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Du bist ein Marketing-Texter fuer Filme."},
                {"role": "user", "content": "Schreibe eine einzige, packende deutsche "
                 "Tagline (max. 10 Woerter) fuer diesen Film:\n" + movie["overview"][:600]},
            ],
            temperature=0.9,
            max_tokens=40,
        )
        return resp.choices[0].message.content.strip().strip('"')
    except Exception:
        return ""
