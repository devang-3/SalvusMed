# SalvusMed

**SalvusMed** is a medicine discovery and safety guide that helps users search medicines by name or symptom, compare information-retrieval algorithms side by side, and find composition-based alternatives (generic substitutes).

> **Disclaimer:** SalvusMed is for educational and informational purposes only. It is not medical advice. Always consult a qualified healthcare professional before starting, stopping, or switching any medicine.

---

## Name & Meaning

**SalvusMed** combines two roots:

| Part | Origin | Meaning |
|------|--------|---------|
| **Salvus** | Latin | *safe*, *saved*, *healthy* |
| **Med** | English (short for *medicine*) | pharmaceutical / therapeutic context |

Together, the name reflects the project's goal: **helping people make safer, more informed medicine choices** — by surfacing substitutes, side effects, and patient review signals in one place.

---

## Why I Built This

Medicine names are often unfamiliar, brand-specific, and hard to compare. When a prescribed drug is unavailable or expensive, patients and caregivers need to know whether an alternative with the **same active composition** exists — but that information is scattered across pharmacy sites and product labels.

I built SalvusMed to address three practical problems:

1. **Discovery** — Search ~11,800 medicines by brand name or by symptom/condition (e.g. *cough*, *bacterial infection*, *hypertension*).
2. **Transparency** — View composition, manufacturer, uses, side effects, and aggregated review percentages in a single detail view.
3. **Alternatives** — Find medicines that share the same normalized salt-and-strength signature, ranked by a composite review score.
4. **Algorithm comparison** — Symptom search runs **BM25 (Okapi)** and **TF-IDF** in parallel so retrieval quality can be compared on real medicine data — useful for learning how classical IR techniques behave outside textbook examples.

This project was developed as part of my **MDW (Mobile & Distributed Web)** coursework, combining information retrieval concepts with a lightweight web application that runs entirely on a local dataset.

---

## Features

- **Search by medicine name** — Prefix and substring matching with review-score tie-breaking
- **Search by symptom / condition** — Dual-engine retrieval with live BM25 vs TF-IDF comparison panels
- **Medicine detail view** — Composition, uses, side effects, manufacturer, and review distribution bars
- **Composition-based alternatives** — Substitutes grouped by parsed salt + strength signature
- **REST API** — JSON endpoints for health checks, search, detail lookup, and alternatives
- **No external database** — Indexes are built in memory from a single CSV at startup

---

## Tech Stack

### Backend

| Technology | Role |
|------------|------|
| **Python 3** | Core language |
| **Flask 3.x** | Web server, routing, Jinja2 template rendering, JSON API |
| **rank-bm25** | BM25Okapi implementation for probabilistic symptom/condition search |
| **scikit-learn** | `TfidfVectorizer` with sublinear TF scaling for TF-IDF symptom search |

### Data & Indexing

| Component | Description |
|-----------|-------------|
| **Medicine_Details.csv** | Source dataset (~11,825 records) with name, composition, uses, side effects, image URL, manufacturer, and review percentages |
| **`search_index.py`** | Loads CSV, builds in-memory `MedicineIndex` with BM25 corpus, TF-IDF matrix, and composition signature map |
| **`composition.py`** | Parses multi-salt compositions (e.g. `Amoxycillin (500mg) + Clavulanic Acid (125mg)`), normalizes salts/strengths, and generates sortable signatures for alternative matching |
| **`preprocess.py`** | Text normalization, stopword filtering, tokenization, uses-field cleanup, side-effect splitting, and review-score weighting |
| **`paths.py`** | Centralized path resolution for the dataset |

### Frontend

| Technology | Role |
|------------|------|
| **HTML5 + Jinja2** | Server-rendered shell (`templates/index.html`) |
| **CSS3** | Custom responsive layout and styling (`static/style.css`) |
| **Vanilla JavaScript** | Debounced search, tab switching (name vs symptom), fetch calls to Flask API, dynamic result/detail rendering (`static/app.js`) |

### Retrieval Pipeline (Symptom Search)

1. User query is tokenized (lowercase, alphanumeric, stopwords removed).
2. **BM25** scores each medicine document built from uses, composition, side effects, and manufacturer.
3. **TF-IDF** transforms the same tokenized corpus and computes cosine-like dot-product scores against the query vector.
4. Results are ranked by score, with `review_score` as a secondary sort key.
5. The UI displays both algorithm panels plus a deduplicated combined list.

### Alternative Matching

1. Composition string is split on `+` into individual components.
2. Each component is parsed into `(salt, strength)` pairs.
3. Pairs are sorted and joined into a canonical signature (e.g. `amoxycillin:500mg|clavulanicacid:125mg`).
4. All medicines sharing the same signature are returned as alternatives, excluding the selected brand.

---

## Project Structure

```
SalvusMed/
├── app.py                 # Flask app, routes, CLI entry point
├── search_index.py        # Dataset load, BM25/TF-IDF indexes, search logic
├── composition.py         # Composition parsing & signature generation
├── preprocess.py          # Tokenization, normalization, review scoring
├── paths.py                 # Dataset path constants
├── Medicine_Details.csv     # Medicine dataset
├── requirements.txt         # Python dependencies
├── run.bat                  # Windows quick-start script
├── templates/
│   └── index.html           # Main UI
└── static/
    ├── style.css
    └── app.js
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Index load status and medicine count |
| `GET` | `/api/search/name?q=` | Search by medicine name |
| `GET` | `/api/search/symptom?q=&engine=bm25\|tfidf` | Symptom search (single engine) |
| `GET` | `/api/search/symptom/compare?q=` | Side-by-side BM25 vs TF-IDF results |
| `GET` | `/api/medicine?name=` | Full detail for one medicine |
| `GET` | `/api/alternatives?name=` | Composition-matched substitutes |

---

## Getting Started

### Prerequisites

- Python 3.10+ recommended
- `pip`

### Installation & Run

```bash
cd SalvusMed
pip install -r requirements.txt
python app.py
```

The server starts at **http://127.0.0.1:5001** and opens in your default browser.

**Windows:** double-click `run.bat` or run it from a terminal.

### CLI Options

```bash
python app.py --host 0.0.0.0 --port 5001 --no-browser
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `5001` | Listen port |
| `--no-browser` | off | Skip auto-opening the browser |

---

## Dependencies

```
flask>=3.0.0
rank-bm25>=0.2.2
scikit-learn>=1.4.0
```

---

## Dataset

`Medicine_Details.csv` contains structured medicine records with the following fields:

- Medicine Name
- Composition
- Uses
- Side_effects
- Image URL
- Manufacturer
- Excellent / Average / Poor Review %

Data is indexed entirely in memory at startup. First launch may take a few seconds while BM25 and TF-IDF matrices are built.

---

## License

This project is intended for educational use. Dataset attribution and licensing should be respected if the CSV is redistributed.
