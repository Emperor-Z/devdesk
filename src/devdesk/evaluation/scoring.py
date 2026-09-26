"""Pure scoring for the eval harness — no I/O, no model calls.

Kept separate from the runner so every metric is unit-testable offline.
Deterministic checks only (string/rank matching against hand-written
expectations): no LLM-as-judge, which would cost free-tier quota and add
its own variance to the thing being measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

AGENTS = ("verisim", "ares", "cross", "none")

# Strings GracefulDegradationPlugin substitutes for a failed model/tool call.
# An answer containing one is an infrastructure failure, not a quality one,
# so it's reported separately rather than silently counted as a wrong answer.
_DEGRADED_MARKERS = ("couldn't reach the model right now", "is temporarily unavailable")

# Any citation of an indexed doc — every tenant's corpus is markdown only.
_CITATION_RE = re.compile(r"[\w./-]+\.md\b", re.IGNORECASE)


@dataclass(frozen=True)
class EvalQuery:
    id: str
    question: str
    expected_agent: str
    expected_sources: tuple[str, ...] = ()
    expected_keywords: tuple[str, ...] = ()


def load_queries(path: Path) -> list[EvalQuery]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    queries: list[EvalQuery] = []
    seen: set[str] = set()
    for item in raw:
        q = EvalQuery(
            id=item["id"],
            question=item["question"],
            expected_agent=item["expected_agent"],
            expected_sources=tuple(item.get("expected_sources") or ()),
            expected_keywords=tuple(item.get("expected_keywords") or ()),
        )
        if q.expected_agent not in AGENTS:
            raise ValueError(f"{q.id}: expected_agent must be one of {AGENTS}")
        if q.id in seen:
            raise ValueError(f"duplicate query id: {q.id}")
        if q.expected_agent != "none" and not q.expected_sources:
            raise ValueError(f"{q.id}: in-scope queries need expected_sources")
        seen.add(q.id)
        queries.append(q)
    return queries


# --- retrieval mode ---------------------------------------------------------


@dataclass
class RetrievalScore:
    id: str
    expected_agent: str
    retrieved: list[str]
    top_similarity: float | None
    first_relevant_rank: int | None  # 1-based; None if no expected source retrieved
    passed: bool

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank else 0.0


def score_retrieval(
    query: EvalQuery, retrieved: list[str], similarities: list[float]
) -> RetrievalScore:
    """`retrieved` is the ranked source_file list the relevant tool would see.

    In-scope: passes if any expected source is retrieved at all (hit@k).
    Out-of-scope: passes only if the similarity cutoff filtered everything —
    i.e. the store itself says "nothing relevant", before any model has a
    chance to paper over it.
    """
    rank = next(
        (i + 1 for i, src in enumerate(retrieved) if src in query.expected_sources), None
    )
    passed = not retrieved if query.expected_agent == "none" else rank is not None
    return RetrievalScore(
        id=query.id,
        expected_agent=query.expected_agent,
        retrieved=retrieved,
        top_similarity=round(max(similarities), 3) if similarities else None,
        first_relevant_rank=rank,
        passed=passed,
    )


# --- end-to-end mode --------------------------------------------------------


@dataclass
class E2EScore:
    id: str
    expected_agent: str
    answer: str
    router_tool_calls: list[str]
    latency_s: float
    degraded: bool
    routing_correct: bool | None  # None = not scored (out-of-scope query)
    cited_expected_source: bool | None
    missing_keywords: list[str] = field(default_factory=list)
    abstained: bool | None = None
    passed: bool = False


def routing_correct(expected_agent: str, calls: list[str]) -> bool | None:
    called = set(calls)
    specialists = {"verisim_agent", "ares_agent"}
    if expected_agent == "none":
        return None
    if expected_agent == "cross":
        return "search_all_projects" in called or specialists <= called
    # Single-project: the right specialist, and no detour to the other one.
    return f"{expected_agent}_agent" in called and not (
        called & (specialists - {f"{expected_agent}_agent"})
    )


def score_e2e(
    query: EvalQuery, answer: str, router_tool_calls: list[str], latency_s: float
) -> E2EScore:
    lowered = answer.lower()
    degraded = not answer.strip() or any(m in lowered for m in _DEGRADED_MARKERS)
    routing = routing_correct(query.expected_agent, router_tool_calls)

    if query.expected_agent == "none":
        abstained = not _CITATION_RE.search(answer)
        return E2EScore(
            id=query.id,
            expected_agent=query.expected_agent,
            answer=answer,
            router_tool_calls=router_tool_calls,
            latency_s=round(latency_s, 2),
            degraded=degraded,
            routing_correct=None,
            cited_expected_source=None,
            abstained=abstained,
            passed=abstained and not degraded,
        )

    cited = any(src.lower() in lowered for src in query.expected_sources)
    missing = [k for k in query.expected_keywords if k.lower() not in lowered]
    return E2EScore(
        id=query.id,
        expected_agent=query.expected_agent,
        answer=answer,
        router_tool_calls=router_tool_calls,
        latency_s=round(latency_s, 2),
        degraded=degraded,
        routing_correct=routing,
        cited_expected_source=cited,
        missing_keywords=missing,
        passed=bool(routing) and cited and not missing and not degraded,
    )


# --- aggregation ------------------------------------------------------------


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize_retrieval(scores: list[RetrievalScore]) -> dict:
    in_scope = [s for s in scores if s.expected_agent != "none"]
    out_scope = [s for s in scores if s.expected_agent == "none"]
    return {
        "n": len(scores),
        "pass_rate": _rate([s.passed for s in scores]),
        "hit_at_k": _rate([s.passed for s in in_scope]),
        "hit_at_1": _rate([s.first_relevant_rank == 1 for s in in_scope]),
        "mrr": round(sum(s.reciprocal_rank for s in in_scope) / len(in_scope), 3)
        if in_scope
        else None,
        "out_of_scope_rejected": _rate([s.passed for s in out_scope]),
    }


def summarize_e2e(scores: list[E2EScore]) -> dict:
    healthy = [s for s in scores if not s.degraded]
    in_scope = [s for s in healthy if s.expected_agent != "none"]
    out_scope = [s for s in healthy if s.expected_agent == "none"]
    latencies = sorted(s.latency_s for s in healthy)
    return {
        "n": len(scores),
        "degraded": len(scores) - len(healthy),
        # Quality rates are over non-degraded runs: a free-tier 503 says
        # nothing about routing or grounding, so it shouldn't move them.
        "pass_rate": _rate([s.passed for s in healthy]),
        "routing_accuracy": _rate([bool(s.routing_correct) for s in in_scope]),
        "citation_accuracy": _rate([bool(s.cited_expected_source) for s in in_scope]),
        "keyword_recall": _rate([not s.missing_keywords for s in in_scope]),
        "abstention_rate": _rate([bool(s.abstained) for s in out_scope]),
        "latency_p50_s": latencies[len(latencies) // 2] if latencies else None,
        "latency_max_s": latencies[-1] if latencies else None,
    }
