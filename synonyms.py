"""Lay-term → clinical synonym groups for query expansion (rule-based, no ML)."""
from __future__ import annotations

import re

# Each key expands to related tokens that may appear in the medicine dataset.
QUERY_SYNONYMS: dict[str, list[str]] = {
    "belly": ["abdominal", "stomach", "gastric"],
    "tummy": ["abdominal", "stomach", "gastric"],
    "stomach": ["abdominal", "gastric"],
    "abdominal": ["stomach", "gastric"],
    "gastric": ["stomach", "abdominal"],
    "head": ["headache"],
    "headache": ["head", "migraine"],
    "migraine": ["headache", "head"],
    "sugar": ["diabetes", "diabetic", "hyperglycemia"],
    "diabetes": ["diabetic", "hyperglycemia"],
    # Do not map to "blood" / "pressure" alone — they appear in almost every record.
    "bp": ["hypertension"],
    "hypertension": ["bp"],
    "fever": ["pyrexia", "hyperthermia", "temperature"],
    "pyrexia": ["fever"],
    "cold": ["cough", "flu", "influenza"],
    "flu": ["influenza", "cold"],
    "influenza": ["flu", "cold"],
    "cough": ["coughing", "dry cough"],
    "heart": ["cardiac", "cardiovascular"],
    "chest": ["thoracic", "respiratory"],
    "skin": ["dermatitis", "dermatological"],
    "itch": ["itching", "pruritus"],
    "itching": ["itch", "pruritus"],
    "allergy": ["allergic", "hypersensitivity"],
    "throat": ["pharyngitis", "sore"],
    "sore": ["throat", "pharyngitis"],
    "ear": ["otic", "aural"],
    "eye": ["ocular"],
    "nose": ["nasal", "rhinitis"],
    "joint": ["arthritis", "rheumatoid", "arthralgia"],
    "muscle": ["muscular", "myalgia"],
    "back": ["spinal", "lumbar"],
    "kidney": ["renal"],
    "liver": ["hepatic"],
    "lung": ["pulmonary", "respiratory"],
    "breath": ["breathing", "respiratory", "asthma"],
    "asthma": ["breathing", "respiratory", "wheezing"],
    "cholesterol": ["hyperlipidemia", "lipid"],
    "acidity": ["acid", "gerd", "ulcer", "reflux"],
    "gas": ["flatulence", "bloating"],
    "bloating": ["gas", "flatulence"],
    "loose": ["diarrhea", "diarrhoea"],
    "diarrhea": ["diarrhoea", "loose"],
    "diarrhoea": ["diarrhea", "loose"],
    "vomit": ["vomiting", "nausea"],
    "vomiting": ["vomit", "nausea"],
    "nausea": ["vomiting", "vomit"],
    "pimple": ["acne"],
    "acne": ["pimple"],
    "worm": ["helminth", "anthelmintic"],
    "pain": ["ache"],
    "ache": ["pain"],
}


_CLAUSE_SPLIT_RE = re.compile(r"[,;]+")


def parse_query_clauses(query: str) -> list[list[str]]:
    """Split on comma/semicolon into clauses; tokenize each (e.g. 'belly pain, sugar bp')."""
    from preprocess import tokenize

    parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(query) if part.strip()]
    if not parts:
        return []
    return [tokenize(part) for part in parts]


def clause_concepts(clause_tokens: list[str]) -> list[set[str]]:
    return query_concepts(clause_tokens)


def clause_matches(med_tokens: set[str], clause_tokens: list[str]) -> bool:
    return all(concept & med_tokens for concept in clause_concepts(clause_tokens))


def query_matches(med_tokens: set[str], clauses: list[list[str]]) -> bool:
    if not clauses:
        return False
    if len(clauses) == 1:
        return clause_matches(med_tokens, clauses[0])
    return any(clause_matches(med_tokens, clause) for clause in clauses)


def clause_coverage(med_tokens: set[str], clauses: list[list[str]]) -> float:
    if not clauses:
        return 0.0
    matched = sum(1 for clause in clauses if clause_matches(med_tokens, clause))
    return matched / len(clauses)


def count_clause_matches(medicines: list, clauses: list[list[str]]) -> list[dict]:
    """Per-clause match counts for stats / UI."""
    counts: list[dict] = []
    for clause_tokens in clauses:
        concepts = clause_concepts(clause_tokens)
        label = " ".join(clause_tokens)
        matched = sum(
            1
            for med in medicines
            if all(concept & set(med.search_tokens) for concept in concepts)
        )
        counts.append({"label": label, "tokens": clause_tokens, "matches": matched})
    return counts


def scoring_tokens_for_clauses(clauses: list[list[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for clause_tokens in clauses:
        for token in scoring_tokens(clause_tokens):
            if token not in seen:
                result.append(token)
                seen.add(token)
    return result


def describe_expansion_for_clauses(clauses: list[list[str]]) -> list[str]:
    return scoring_tokens_for_clauses(clauses)


def query_concepts(tokens: list[str]) -> list[set[str]]:
    """One concept per query token; each concept includes synonyms."""
    concepts: list[set[str]] = []
    for token in tokens:
        group = {token}
        group.update(QUERY_SYNONYMS.get(token, []))
        concepts.append(group)
    return concepts


def scoring_tokens(tokens: list[str]) -> list[str]:
    """Flat token list for BM25 / TF-IDF / cosine scoring."""
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for candidate in [token] + QUERY_SYNONYMS.get(token, []):
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
    return result


def describe_expansion(tokens: list[str]) -> list[str]:
    """Human-readable expansion for API / UI."""
    if not tokens:
        return []
    expanded = scoring_tokens(tokens)
    if expanded == tokens:
        return tokens
    return expanded


def flatten_clause_tokens(clauses: list[list[str]]) -> list[str]:
    flat: list[str] = []
    for clause in clauses:
        flat.extend(clause)
    return flat
