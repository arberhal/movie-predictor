# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title: Film-Erfolgsprognose (Movie Rating Prediction & Explanation)
- Student: «dein Name»
- GitHub repository URL: «https://github.com/dein-account/movie-success»
- Deployment URL: https://huggingface.co/spaces/haljiarb/film-predictor
- Submission date: «TT.MM.2026»

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [ ] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [ ] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: ML Numeric Data
- Primary block 2: NLP

If a third block is selected, it is documented and graded separately as extra work.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Vor dem Kinostart ist unklar, wie ein Film beim Publikum ankommt. Aus den Eckdaten eines Films (Budget, Laufzeit, Genres) und der Handlung soll die zu erwartende Zuschauerbewertung geschätzt und verständlich erklärt werden.
- Goal: Vorhersage der TMDB-Zuschauerbewertung (`vote_average`, 0–10) plus eine natürlichsprachliche Begründung und eine Tagline.
- Success criteria: (1) deutlich besser als eine Mittelwert-Baseline, (2) die NLP-Textmerkmale verbessern das Modell messbar, (3) lauffähige, deployte App.

### 1.2 Integration Logic
- How the selected blocks interact: Der Handlungstext wird vom NLP-Block per TF-IDF in numerische Merkmale umgewandelt; diese fließen zusammen mit den strukturierten Merkmalen in das ML-Modell. Das ML-Modell liefert eine Vorhersage plus die wichtigsten Einflussfaktoren; daraus erzeugt der NLP-Block (LLM) die Erklärung und eine Tagline.
- Data and output flow between blocks:
  `Eingabe (Eckdaten + Handlung)` → `NLP: TF-IDF(Handlung)` → `ML: Vorhersage + Faktoren` → `NLP: LLM-Erklärung + Tagline` → `Ausgabe`.
  Siehe Pipeline in [`src/train.py`](src/train.py#L70-L120) und die Verbindung in der App [`app.py`](app.py#L37-L66).

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | TMDB 5000 Movies (The Movie Database, via GitHub-Mirror) | Strukturierte/tabellarische CSV | ~4.800 Filme, 20 Spalten | Training & Zielvariable (`vote_average`) |
| 2 | `data/recent_movies.csv` (kuratiert, IMDb-Bewertungen) | Tabellarische CSV | 36 Filme (2022–2024) | Unabhängige Out-of-Sample-Evaluation |

*Hinweis zur Quellen-Diversität: Die beiden Quellen haben unterschiedliche Herkunft, Plattform und Erhebungszeitraum — Quelle 1 stammt aus der TMDB-Datenbank (bis ~2017), Quelle 2 ist eine eigenständig kuratierte Liste mit Bewertungen der separaten Plattform IMDb und umfasst neuere Filme (2022–2024).*

#### 2A.2 Preprocessing and Features
- Cleaning steps: Filme mit `vote_count < 10` entfernt (unzuverlässige Bewertung), Zeilen ohne Handlung verworfen, fehlende `runtime` mit Median ersetzt, `TotalCharges`-artige Nullwerte bei Budget als „fehlend" markiert. Siehe [`src/train.py`](src/train.py#L34-L52).
- Preprocessing steps: `budget_log = log1p(budget)`, Flag `has_budget`, `year` aus `release_date`, Genres aus JSON geparst und als Multi-Hot kodiert (`CountVectorizer`).
- Feature engineering and selection: Bewusst **kein** `popularity`/`vote_count` als Merkmal (Daten-Leakage, da erst nach Release verfügbar). Genutzte Merkmale: `budget_log`, `has_budget`, `runtime`, `year`, Genre-Multi-Hot, sowie TF-IDF-Merkmale aus der Handlung (NLP-Block).

#### 2A.3 Model Selection
- Models tested: Ridge-Regression (lineares Baseline-Modell) und RandomForestRegressor (nichtlinear).
- Why these models were chosen: Ridge als einfaches, interpretierbares Referenzmodell; RandomForest, weil er nichtlineare Zusammenhänge und gemischte (numerische + spärliche Text-/Genre-) Merkmale gut verarbeitet.

#### 2A.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Referenzwert | Mittelwert vorhersagen | DummyRegressor | MAE 0.71 / R² 0.00 | — |
| 2 | Strukturierte Merkmale | Budget, Laufzeit, Jahr, Genres | Ridge / RandomForest | RF: MAE 0.575 / R² 0.325 | klar besser als Baseline |
| 3 | Handlungstext ergänzen | + TF-IDF-Merkmale (NLP) | RandomForest (final) | MAE 0.57 / RMSE 0.73 / R² 0.34 | Text verbessert RF (R² 0.325 → 0.341) |

Vergleich siehe [`src/train.py`](src/train.py#L96-L110).

#### 2A.5 Evaluation and Error Analysis
- Metrics used: MAE, RMSE, R²; 80/20 Train-/Test-Split (`random_state=42`).
- Final results: Bestes Modell RandomForest mit Text — **MAE 0.57, RMSE 0.73, R² 0.34** (Testset, n=879). Out-of-Sample auf der zweiten Quelle (36 neue Filme): **MAE 0.92, R² 0.07**. Siehe [`src/evaluate_external.py`](src/evaluate_external.py) und `docs/figures/05_external_eval.png`.
- Error patterns and likely causes: Das Modell schätzt fast alles in den Bereich 6–7; es unterschätzt herausragende Filme (z. B. Dune: Part Two, Oppenheimer) und überschätzt Flops (z. B. Madame Web, Borderlands). Ursachen: einzelne Merkmale tragen nur schwaches Signal (Genre-Spannweite < 1 Punkt, siehe `docs/figures/02_rating_by_genre.png`), Verteilungsverschiebung zu neueren Filmen, IMDb- vs. TMDB-Skala, und die grundsätzliche Schwierigkeit, subjektive Bewertungen aus Vorab-Merkmalen vorherzusagen.

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s): TF-IDF-Merkmale aus dem Handlungstext (NLP-Block).
- Outputs provided to other block(s): vorhergesagte Bewertung + wichtigste Einflussfaktoren (an den NLP-Block zur Erklärung).

Guidance hint: Keep entries practical and evidence-based.
Evidence hint: Add values, not only claims.

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Handlungstexte (`overview`) aus TMDB 5000 | Freitext | ~4.800 Texte | TF-IDF-Merkmale fürs ML-Modell |
| 2 | Nutzereingabe in der App (Titel + Handlung) | Freitext | pro Anfrage | Eingabe für die LLM-Erklärung |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing: TF-IDF mit englischen Stoppwörtern, `max_features=300`, `min_df=5`, 1- und 2-Gramme. Siehe [`src/train.py`](src/train.py#L86-L92).
- Prompt design or retrieval setup: Strukturierter Prompt, der dem LLM die Vorhersage **und** verankerte Faktoren (Laufzeit vs. Median, Genre-Durchschnitte, Budget) übergibt, mit der Anweisung, sich nur auf diese Fakten zu beziehen (keine Halluzination). Siehe [`nlp.py`](nlp.py#L26-L95).

#### 2B.3 Approach Selection
- Approach used: Kombination aus **klassischem NLP** (TF-IDF als Merkmale fürs ML-Modell) und **Prompt Engineering** mit einem LLM (OpenAI `gpt-4o-mini`) für Erklärung und Tagline.
- Alternatives considered: Transformer-Embeddings (rechenintensiver, für diesen Zweck nicht nötig); reine LLM-Bewertung (weniger kontrollierbar, kein quantitatives Modell).

#### 2B.4 Comparison and Iterations
| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Funktionsfähige Erklärung | regelbasierte Vorlage | Fallback ohne LLM | sachlich, aber starr | — |
| 2 | Natürliche Sprache | knapper 2-Satz-Prompt | gpt-4o-mini | lesbarer, flüssiger | verständlicher |
| 3 | Verankerung + Grenzen | Faktoren + 3–4 Sätze inkl. Einschränkung | gpt-4o-mini | korrekt, nennt Unsicherheit | weniger Halluzination, informativer |

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy: Qualitativ (Spiegelt die Erklärung die Modellfaktoren? Keine erfundenen Aussagen?) plus quantitativer Nachweis über die ML-Ablation: Die Text-Merkmale verbessern R² messbar (0.325 → 0.341).
- Results: Der verankerte „detailed"-Prompt liefert zutreffende, gut lesbare Erklärungen; die Tagline-Funktion erzeugt passende Slogans; ohne API-Key greift ein regelbasierter Fallback.
- Error patterns and likely causes: Gelegentlich generische Formulierungen; das LLM neigt dazu, kleine Unterschiede überzubetonen, da das ML-Modell selbst wenig differenziert.

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s): Vorhersage + Einflussfaktoren vom ML-Block.
- Outputs provided to other block(s): (a) TF-IDF-Merkmale an den ML-Block, (b) Erklärung + Tagline an die Nutzeroberfläche.

Guidance hint: Show concrete prompt or retrieval decisions.
Evidence hint: Include representative outputs or failure cases.

### 2C. Computer Vision (If selected)

N/A – nicht ausgewählt.

#### 2C.1 Data Source(s)
N/A

#### 2C.2 Preprocessing and Augmentation
N/A

#### 2C.3 Model Selection
N/A

#### 2C.4 Model Comparison and Iterations
N/A

#### 2C.5 Evaluation and Error Analysis
N/A

#### 2C.6 Integration with Other Block(s)
N/A

---

## 3. Deployment

- Deployment URL: https://huggingface.co/spaces/haljiarb/film-predictor
- Main user flow: Tab 1 „Eigenen Film bewerten" — Budget, Laufzeit, Jahr, Genres und Handlung eingeben → vorhergesagte Bewertung + LLM-Erklärung + Tagline. Tab 2 „Test an echten Filmen" — einen bekannten Film (2022–2024) wählen → Vergleich Modell-Vorhersage vs. echte IMDb-Bewertung (zweite Datenquelle).
- Screenshot or short demo: App-Screenshots beider Tabs unter `docs/figures/app_tab1.png` (Eigenen Film bewerten) und `docs/figures/app_tab2.png` (Test an echten Filmen); zusätzlich die Analyse-Plots in `docs/figures/`.

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- Environment setup: `pip install -r requirements.txt` (Python 3.11+).
- Data setup: `data/tmdb_5000_movies.csv` und `data/recent_movies.csv` liegen im Repo; fehlt das Trainings-CSV, lädt [`model_utils.py`](model_utils.py) es automatisch von GitHub.
- Training command(s): `python src/train.py` (erzeugt `models/rating_model.joblib` und `models/meta.json`).
- Inference/run command(s): `export OPENAI_API_KEY="sk-..."` und `python app.py` (App unter `http://localhost:7860`).
- Reproducibility notes: Feste Zufallszahl `random_state=42`; Paketversionen in `requirements.txt`. Hinweis: Bei abweichender scikit-learn-Version `python src/train.py` neu ausführen, damit das gespeicherte Modell zur Umgebung passt. EDA: `python src/eda.py`; Out-of-Sample-Test: `python src/evaluate_external.py`.

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [ ] Third selected block implemented with strong quality
- [ ] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [x] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

Evidence for selected bonus items:
- **Extended evaluation:** Zusätzlich zum Standard-Train-/Test-Split wurde das Modell auf einer **unabhängigen zweiten Datenquelle** (36 reale Filme 2022–2024) out-of-sample geprüft. Das deckt eine Generalisierungslücke auf (MAE 0.57 → 0.92, R² 0.34 → 0.07) und führt zu einer fundierten Fehleranalyse inklusive Diskussion der Ursachen (schwache Einzelmerkmale, Verteilungsverschiebung, IMDb-/TMDB-Skalenunterschied). Code: [`src/evaluate_external.py`](src/evaluate_external.py), Visualisierung: `docs/figures/05_external_eval.png`.
