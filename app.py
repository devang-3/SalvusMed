"""
SalvusMed — BM25 vs TF-IDF vs BoW cosine symptom search + composition alternatives.

Usage:
    cd SalvusMed
    pip install -r requirements.txt
    python app.py

Open http://127.0.0.1:5001
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from search_index import MedicineIndex

APP_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)
index: MedicineIndex | None = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    loaded = index is not None
    count = len(index.medicines) if loaded else 0
    return jsonify({"ok": loaded, "medicines": count})


@app.route("/api/search/name")
def api_search_name():
    if index is None:
        return jsonify({"error": "Index not loaded"}), 503

    query = request.args.get("q", "")
    limit = min(int(request.args.get("limit", 12)), 30)
    if not query.strip():
        return jsonify({"query": "", "results": []})

    results = index.search_by_name(query, limit=limit)
    return jsonify({"query": query, "results": results})


@app.route("/api/search/symptom")
def api_search_symptom():
    if index is None:
        return jsonify({"error": "Index not loaded"}), 503

    query = request.args.get("q", "")
    limit = min(int(request.args.get("limit", 15)), 30)
    if not query.strip():
        return jsonify({"query": "", "results": []})

    engine = request.args.get("engine", "bm25").strip().lower()
    if engine == "tfidf":
        results = index.search_by_symptom_tfidf(query, limit=limit)
    elif engine == "cosine":
        results = index.search_by_symptom_cosine(query, limit=limit)
    else:
        results = index.search_by_symptom_bm25(query, limit=limit)
    return jsonify({"query": query, "engine": engine, "results": results})


@app.route("/api/search/symptom/compare")
def api_search_symptom_compare():
    if index is None:
        return jsonify({"error": "Index not loaded"}), 503

    query = request.args.get("q", "")
    per_algo = min(int(request.args.get("per_algo", 8)), 15)
    combined_limit = min(int(request.args.get("combined", 12)), 20)
    if not query.strip():
        return jsonify({"query": "", "combined": [], "algorithms": {}, "stats": {}})

    return jsonify(
        index.search_symptom_compare(
            query,
            per_algo=per_algo,
            combined_limit=combined_limit,
        )
    )


@app.route("/api/medicine")
def api_medicine():
    if index is None:
        return jsonify({"error": "Index not loaded"}), 503

    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "name parameter required"}), 400

    med = index.lookup(name)
    if not med:
        return jsonify({"found": False, "name": name}), 404

    return jsonify({"found": True, "medicine": index.to_detail(med)})


@app.route("/api/alternatives")
def api_alternatives():
    if index is None:
        return jsonify({"error": "Index not loaded"}), 503

    name = request.args.get("name", "").strip()
    limit = min(int(request.args.get("limit", 20)), 50)
    if not name:
        return jsonify({"error": "name parameter required"}), 400

    return jsonify(index.find_alternatives(name, limit=limit))


def main() -> None:
    global index

    parser = argparse.ArgumentParser(description="SalvusMed medicine search UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("Loading Medicine_Details.csv and building BM25 + TF-IDF + BoW indexes...")
    index = MedicineIndex.build()
    print(f"Ready: {len(index.medicines):,} medicines indexed.")

    url = f"http://{args.host}:{args.port}"
    print(f"Open {url}")
    if not args.no_browser:
        webbrowser.open(url)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
