"""Eval runner: python eval/run_eval.py [--mode retrieval|e2e] [--ids ...]

retrieval — calls the real `search_docs` / `search_all_projects` tools
            directly (one embedding call per query, no chat-model calls),
            scoring whether the expected source reaches the agent at all.
e2e       — runs each question through the full router -> specialist loop
            on a single shared runner (so RateLimitPlugin paces the whole
            batch) and scores routing, citations, keywords and abstention.

Writes a JSON result file to eval/results/ and prints a summary table.
`--fail-under` turns the pass rate into an exit code for CI gating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from devdesk import config
from devdesk.evaluation.scoring import (
    E2EScore,
    EvalQuery,
    RetrievalScore,
    load_queries,
    score_e2e,
    score_retrieval,
    summarize_e2e,
    summarize_retrieval,
)

EVAL_DIR = config.REPO_ROOT / "eval"
DEFAULT_QUERIES = EVAL_DIR / "queries.yaml"
RESULTS_DIR = EVAL_DIR / "results"


def run_retrieval(queries: list[EvalQuery]) -> list[RetrievalScore]:
    from devdesk.tools.search_all_projects import search_all_projects
    from devdesk.tools.search_docs import search_docs

    scores = []
    for q in queries:
        if q.expected_agent in ("cross", "none"):
            hits = search_all_projects(q.question)["hits"]
        else:
            hits = search_docs(project=q.expected_agent, query=q.question)["hits"]
        scores.append(
            score_retrieval(
                q,
                retrieved=[h["source_file"] for h in hits],
                similarities=[h["similarity"] for h in hits],
            )
        )
    return scores


async def run_e2e(queries: list[EvalQuery]) -> list[E2EScore]:
    from devdesk.cli import build_runner, run_question

    runner = build_runner()
    scores = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q.id} ...", file=sys.stderr, flush=True)
        start = time.monotonic()
        try:
            result = await run_question(runner, q.question)
            answer, calls = result.answer, result.router_tool_calls
            unverified = result.unverified_citations
        except Exception as exc:  # noqa: BLE001 — one bad query mustn't kill the batch
            answer, calls = f"(harness error: {type(exc).__name__}: {exc})", []
            unverified = []
        scores.append(score_e2e(q, answer, calls, time.monotonic() - start, unverified))
    return scores


def _print_table(mode: str, scores: list, summary: dict) -> None:
    print(f"\n## DevDesk eval — {mode} ({config.GEMINI_MODEL})\n")
    if mode == "retrieval":
        print("| id | pass | rank | top sim | retrieved |")
        print("|---|---|---|---|---|")
        for s in scores:
            print(
                f"| {s.id} | {'✅' if s.passed else '❌'} | {s.first_relevant_rank or '-'} "
                f"| {s.top_similarity if s.top_similarity is not None else '-'} "
                f"| {', '.join(dict.fromkeys(s.retrieved)) or '(none)'} |"
            )
    else:
        print("| id | pass | routed | cited | missing keywords | latency |")
        print("|---|---|---|---|---|---|")
        for s in scores:
            status = "⚠️ degraded" if s.degraded else ("✅" if s.passed else "❌")
            cited = "-" if s.cited_expected_source is None else ("✅" if s.cited_expected_source else "❌")
            if s.abstained is not None:
                cited = "abstained" if s.abstained else "cited (should abstain)"
            print(
                f"| {s.id} | {status} | {', '.join(s.router_tool_calls) or '(none)'} "
                f"| {cited} | {', '.join(s.missing_keywords) or '-'} | {s.latency_s}s |"
            )
    print("\n" + "\n".join(f"- **{k}**: {v}" for k, v in summary.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("retrieval", "e2e"), default="retrieval")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--ids", nargs="+", help="only run these query ids")
    parser.add_argument(
        "--split", choices=("dev", "holdout", "all"), default="all", help="query split to run"
    )
    parser.add_argument(
        "--fail-under", type=float, default=0.0, help="exit 1 if pass_rate is below this"
    )
    parser.add_argument("--no-save", action="store_true", help="don't write a result file")
    args = parser.parse_args(argv)

    queries = load_queries(args.queries)
    if args.ids:
        unknown = set(args.ids) - {q.id for q in queries}
        if unknown:
            parser.error(f"unknown query ids: {', '.join(sorted(unknown))}")
        queries = [q for q in queries if q.id in args.ids]
    if args.split != "all":
        queries = [q for q in queries if q.split == args.split]

    if args.mode == "retrieval":
        scores: list = run_retrieval(queries)
        summarize = summarize_retrieval
    else:
        scores = asyncio.run(run_e2e(queries))
        summarize = summarize_e2e
    summary = summarize(scores)
    split_of = {q.id: q.split for q in queries}
    # Per-split breakdown: a prompt change that lifts dev but not holdout is
    # overfitting to the dev questions.
    for split in ("dev", "holdout"):
        subset = [s for s in scores if split_of[s.id] == split]
        if subset and len(subset) < len(scores):
            summary[f"{split}_pass_rate"] = summarize(subset)["pass_rate"]

    _print_table(args.mode, scores, summary)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = RESULTS_DIR / f"{stamp}_{args.mode}.json"
        out.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "model": config.GEMINI_MODEL,
                    "embedder": config.GEMINI_EMBEDDING_MODEL
                    if config.use_gemini_embeddings()
                    else "all-MiniLM-L6-v2",
                    "min_query_similarity": config.min_query_similarity(),
                    "timestamp": stamp,
                    "summary": summary,
                    "results": [asdict(s) for s in scores],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved {out.relative_to(config.REPO_ROOT)}")

    pass_rate = summary["pass_rate"] or 0.0
    return 1 if pass_rate < args.fail_under else 0
