#!/usr/bin/env python3
"""Evaluate retrieval quality against a JSONL query set."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_queries(path: Path) -> List[Dict]:
    queries: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.append(json.loads(line))
    return queries


def result_title(result: Dict) -> str:
    payload = result.get("payload", {})
    return str(payload.get("title", ""))


def result_text(result: Dict) -> str:
    payload = result.get("payload", {})
    return str(payload.get("text", ""))


def fetch_results_via_api(api_base: str, query: str, top_k: int, language: str) -> List[Dict]:
    response = requests.get(
        f"{api_base.rstrip('/')}/search",
        params={"q": query, "top_k": top_k, "language": language},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    results = []
    for item in payload.get("results", []):
        results.append({
            "text": item.get("text", ""),
            "score": item.get("score", 0.0),
            "payload": item.get("payload", {}),
        })
    return results


def matches_expected_title(result: Dict, expected_titles: List[str]) -> bool:
    if not expected_titles:
        return False
    haystack = normalize(result_title(result))
    return any(normalize(title) in haystack for title in expected_titles)


def matches_must_have(result: Dict, must_have: List[str]) -> int:
    haystack = normalize(result_title(result) + " " + result_text(result))
    return sum(1 for term in must_have if normalize(term) in haystack)


def is_relevant(result: Dict, query_spec: Dict) -> bool:
    expected_titles = query_spec.get("expected_titles", []) or []
    must_have = query_spec.get("must_have", []) or []
    if expected_titles and matches_expected_title(result, expected_titles):
        return True
    if must_have:
        return matches_must_have(result, must_have) == len(must_have)
    return False


@dataclass
class QueryResult:
    query: str
    language: str
    relevant_at_5: bool
    relevant_at_10: bool
    top_title: str
    top_score: float
    concept_coverage: float
    matched_terms: List[str]
    expected_titles: List[str]
    must_have: List[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against JSONL queries")
    parser.add_argument("--queries", default=str(REPO_DIR / "eval" / "queries.jsonl"), help="path to JSONL query set")
    parser.add_argument("--top-k", type=int, default=10, help="maximum results to inspect per query")
    parser.add_argument("--output-dir", default=str(REPO_DIR / "reports"), help="directory to write reports")
    parser.add_argument(
        "--backend",
        choices=["api", "direct"],
        default="api",
        help="how to fetch retrieval results",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("API_BASE", "http://127.0.0.1:8000"),
        help="base URL for the FastAPI server when backend=api",
    )
    args = parser.parse_args()

    query_path = Path(args.queries).expanduser().resolve()
    if not query_path.exists():
        raise SystemExit(f"Query file not found: {query_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = load_queries(query_path)
    if not specs:
        raise SystemExit("No queries found")

    per_query: List[QueryResult] = []
    fail_details: List[Dict] = []

    for spec in specs:
        query = str(spec.get("query", "")).strip()
        language = str(spec.get("language", "id")).strip() or "id"
        must_have = [str(term).strip() for term in spec.get("must_have", []) if str(term).strip()]
        expected_titles = [str(title).strip() for title in spec.get("expected_titles", []) if str(title).strip()]

        try:
            if args.backend == "api":
                results = fetch_results_via_api(args.api_base, query, args.top_k, language)
            else:
                from retriever import retrieve  # noqa: E402

                results = retrieve(query=query, top_k=args.top_k, language=language)
        except requests.RequestException as exc:
            raise SystemExit(
                f"Unable to reach API at {args.api_base}. Start `bash rag/run_api.sh` "
                f"or rerun with `--backend direct`. Details: {exc}"
            ) from exc
        top_5 = results[:5]

        hit_5 = any(is_relevant(result, spec) for result in top_5)
        hit_10 = any(is_relevant(result, spec) for result in results[:10])
        top_result = results[0] if results else {}
        top_title = result_title(top_result)
        top_score = float(top_result.get("score", 0.0) or 0.0)

        if must_have:
            matched_terms = [term for term in must_have if normalize(term) in normalize(top_title + " " + result_text(top_result))]
            concept_coverage = len(matched_terms) / len(must_have)
        else:
            matched_terms = []
            concept_coverage = 0.0

        qr = QueryResult(
            query=query,
            language=language,
            relevant_at_5=hit_5,
            relevant_at_10=hit_10,
            top_title=top_title,
            top_score=round(top_score, 4),
            concept_coverage=round(concept_coverage, 4),
            matched_terms=matched_terms,
            expected_titles=expected_titles,
            must_have=must_have,
        )
        per_query.append(qr)

        if not hit_10:
            fail_details.append({
                "query": query,
                "language": language,
                "top_title": top_title,
                "top_score": round(top_score, 4),
                "must_have": must_have,
                "expected_titles": expected_titles,
                "results": [
                    {
                        "title": result_title(r),
                        "score": round(float(r.get("score", 0.0) or 0.0), 4),
                        "book_id": r.get("payload", {}).get("book_id"),
                        "json_path": r.get("payload", {}).get("json_path"),
                    }
                    for r in results[:10]
                ],
            })

    recall_at_5 = sum(1 for item in per_query if item.relevant_at_5) / len(per_query)
    recall_at_10 = sum(1 for item in per_query if item.relevant_at_10) / len(per_query)
    avg_concept_coverage = sum(item.concept_coverage for item in per_query) / len(per_query)
    failed_queries = len([item for item in per_query if not item.relevant_at_10])

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_file": str(query_path),
        "total_queries": len(per_query),
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_10": round(recall_at_10, 4),
        "avg_concept_coverage": round(avg_concept_coverage, 4),
        "failed_queries": failed_queries,
        "queries": [asdict(item) for item in per_query],
        "fail_details": fail_details,
    }

    report_path = output_dir / f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Query file        : {query_path}")
    print(f"Backend           : {args.backend}")
    print(f"Total queries     : {len(per_query)}")
    print(f"Recall@5          : {report['recall_at_5']:.4f}")
    print(f"Recall@10         : {report['recall_at_10']:.4f}")
    print(f"Avg concept cover : {report['avg_concept_coverage']:.4f}")
    print(f"Failed queries    : {failed_queries}")
    print(f"Report path       : {report_path}")


if __name__ == "__main__":
    main()
