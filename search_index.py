"""Load dataset, BM25 + TF-IDF symptom search, name search, composition alternatives."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from composition import composition_signature
from paths import DATASET_PATH
from preprocess import (
    normalize_uses,
    review_score,
    split_side_effects,
    tokenize,
)


@dataclass
class Medicine:
    idx: int
    name: str
    composition: str
    uses: str
    uses_display: str
    side_effects: str
    side_effects_list: list[str]
    image_url: str
    manufacturer: str
    excellent_pct: int
    average_pct: int
    poor_pct: int
    review_score: int
    signature: str
    search_tokens: list[str]


@dataclass
class MedicineIndex:
    medicines: list[Medicine] = field(default_factory=list)
    by_name_lower: dict[str, Medicine] = field(default_factory=dict)
    by_signature: dict[str, list[Medicine]] = field(default_factory=dict)
    names_lower: list[str] = field(default_factory=list)
    bm25_corpus: list[list[str]] = field(default_factory=list)
    bm25: BM25Okapi | None = None
    tfidf_vectorizer: TfidfVectorizer | None = None
    tfidf_matrix: object | None = None

    @classmethod
    def build(cls, csv_path: Path | None = None) -> "MedicineIndex":
        path = csv_path or DATASET_PATH
        medicines: list[Medicine] = []
        by_name_lower: dict[str, Medicine] = {}
        by_signature: dict[str, list[Medicine]] = {}
        bm25_corpus: list[list[str]] = []

        with open(path, newline="", encoding="utf-8") as handle:
            for idx, row in enumerate(csv.DictReader(handle)):
                name = (row.get("Medicine Name") or "").strip()
                if not name:
                    continue

                composition = (row.get("Composition") or "").strip()
                uses_raw = (row.get("Uses") or "").strip()
                uses_display = normalize_uses(uses_raw)
                side_effects = (row.get("Side_effects") or "").strip()
                excellent = int(row.get("Excellent Review %") or 0)
                average = int(row.get("Average Review %") or 0)
                poor = int(row.get("Poor Review %") or 0)
                signature = composition_signature(composition)

                search_text = " ".join(
                    [
                        uses_display,
                        composition,
                        side_effects,
                        (row.get("Manufacturer") or "").strip(),
                    ]
                )
                tokens = tokenize(search_text)

                med = Medicine(
                    idx=idx,
                    name=name,
                    composition=composition,
                    uses=uses_raw,
                    uses_display=uses_display,
                    side_effects=side_effects,
                    side_effects_list=split_side_effects(side_effects),
                    image_url=(row.get("Image URL") or "").strip(),
                    manufacturer=(row.get("Manufacturer") or "").strip(),
                    excellent_pct=excellent,
                    average_pct=average,
                    poor_pct=poor,
                    review_score=review_score(excellent, average, poor),
                    signature=signature,
                    search_tokens=tokens,
                )
                medicines.append(med)
                by_name_lower[name.lower()] = med
                bm25_corpus.append(tokens)
                if signature:
                    by_signature.setdefault(signature, []).append(med)

        bm25 = BM25Okapi(bm25_corpus) if bm25_corpus else None
        tfidf_vectorizer = None
        tfidf_matrix = None
        if bm25_corpus:
            joined_corpus = [" ".join(tokens) for tokens in bm25_corpus]
            tfidf_vectorizer = TfidfVectorizer(
                analyzer=lambda text: text.split(),
                lowercase=False,
                sublinear_tf=True,
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform(joined_corpus)

        return cls(
            medicines=medicines,
            by_name_lower=by_name_lower,
            by_signature=by_signature,
            names_lower=[m.name.lower() for m in medicines],
            bm25_corpus=bm25_corpus,
            bm25=bm25,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
        )

    def lookup(self, name: str) -> Medicine | None:
        return self.by_name_lower.get(name.strip().lower())

    @staticmethod
    def _summary(
        med: Medicine,
        score: float | None = None,
        source: str | None = None,
    ) -> dict:
        item = {
            "name": med.name,
            "composition": med.composition,
            "manufacturer": med.manufacturer,
            "uses": med.uses_display,
            "image_url": med.image_url,
            "excellent_pct": med.excellent_pct,
            "average_pct": med.average_pct,
            "poor_pct": med.poor_pct,
            "review_score": med.review_score,
            "side_effects_count": len(med.side_effects_list),
        }
        if score is not None:
            item["score"] = round(float(score), 4)
        if source is not None:
            item["source"] = source
        return item

    def _rank_scores(
        self,
        scores: list[float] | object,
        source: str,
        limit: int,
        min_score: float = 1e-9,
    ) -> list[dict]:
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-item[1], -self.medicines[item[0]].review_score),
        )
        results: list[dict] = []
        for idx, score in ranked:
            if score <= min_score:
                continue
            med = self.medicines[idx]
            results.append(self._summary(med, score, source))
            if len(results) >= limit:
                break
        return results

    def to_detail(self, med: Medicine) -> dict:
        return {
            **self._summary(med),
            "uses_raw": med.uses,
            "side_effects": med.side_effects,
            "side_effects_list": med.side_effects_list,
        }

    def search_by_name(self, query: str, limit: int = 12) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []

        scored: list[tuple[float, Medicine]] = []
        for med in self.medicines:
            name_lower = med.name.lower()
            if q == name_lower:
                scored.append((100.0, med))
            elif name_lower.startswith(q):
                scored.append((80.0 - len(name_lower) * 0.01, med))
            elif q in name_lower:
                scored.append((50.0 - name_lower.index(q) * 0.05, med))

        scored.sort(key=lambda item: (-item[0], -item[1].review_score, item[1].name))
        seen: set[str] = set()
        results: list[dict] = []
        for score, med in scored:
            key = med.name.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(self._summary(med, score))
            if len(results) >= limit:
                break
        return results

    def search_by_symptom_bm25(self, query: str, limit: int = 15) -> list[dict]:
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        return self._rank_scores(
            self.bm25.get_scores(tokens),
            source="bm25",
            limit=limit,
            min_score=0.0,
        )

    def search_by_symptom_tfidf(self, query: str, limit: int = 15) -> list[dict]:
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        query_vec = self.tfidf_vectorizer.transform([" ".join(tokens)])
        scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
        return self._rank_scores(scores, source="tfidf", limit=limit)

    def search_by_symptom(self, query: str, limit: int = 15) -> list[dict]:
        return self.search_by_symptom_bm25(query, limit=limit)

    def search_symptom_compare(
        self,
        query: str,
        per_algo: int = 8,
        combined_limit: int = 12,
    ) -> dict:
        bm25_results = self.search_by_symptom_bm25(query, limit=per_algo)
        tfidf_results = self.search_by_symptom_tfidf(query, limit=per_algo)

        combined: list[dict] = []
        seen: set[str] = set()
        for item in bm25_results + tfidf_results:
            key = item["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            combined.append(item)
            if len(combined) >= combined_limit:
                break

        max_hits = max(len(bm25_results), len(tfidf_results), 1)
        return {
            "query": query,
            "combined": combined,
            "algorithms": {
                "bm25": bm25_results,
                "tfidf": tfidf_results,
            },
            "stats": {
                "bm25": len(bm25_results),
                "tfidf": len(tfidf_results),
                "max_hits": max_hits,
            },
        }

    def find_alternatives(self, name: str, limit: int = 20) -> dict:
        selected = self.lookup(name)
        if not selected:
            return {
                "found": False,
                "selected": name,
                "composition": "",
                "alternatives": [],
                "total_alternatives": 0,
            }

        if not selected.signature:
            return {
                "found": True,
                "selected": selected.name,
                "composition": selected.composition,
                "alternatives": [],
                "total_alternatives": 0,
            }

        alts = [
            med
            for med in self.by_signature.get(selected.signature, [])
            if med.name.lower() != selected.name.lower()
        ]
        alts.sort(
            key=lambda med: (-med.review_score, -med.excellent_pct, med.name.lower())
        )

        return {
            "found": True,
            "selected": selected.name,
            "composition": selected.composition,
            "signature": selected.signature,
            "alternatives": [self._summary(med) for med in alts[:limit]],
            "total_alternatives": len(alts),
        }
