from pathlib import Path

import pytest

from devdesk.evaluation.scoring import (
    EvalQuery,
    load_queries,
    routing_correct,
    score_e2e,
    score_retrieval,
    summarize_e2e,
    summarize_retrieval,
)

REPO_QUERIES = Path(__file__).resolve().parents[1] / "eval" / "queries.yaml"

_VERISIM = EvalQuery(
    id="v",
    question="q",
    expected_agent="verisim",
    expected_sources=("docs/consent_gate.md",),
    expected_keywords=("trust",),
)
_NONE = EvalQuery(id="n", question="q", expected_agent="none")


def test_repo_query_set_loads_and_is_valid():
    queries = load_queries(REPO_QUERIES)
    assert len(queries) >= 10
    assert {q.expected_agent for q in queries} == {"verisim", "ares", "cross", "none"}


def test_load_queries_rejects_in_scope_query_without_sources(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text("- {id: x, question: q, expected_agent: ares}\n")
    with pytest.raises(ValueError, match="expected_sources"):
        load_queries(bad)


def test_load_queries_rejects_duplicate_ids(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text(
        "- {id: x, question: q, expected_agent: none}\n"
        "- {id: x, question: q, expected_agent: none}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_queries(bad)


def test_retrieval_records_rank_of_first_expected_source():
    s = score_retrieval(_VERISIM, ["README.md", "docs/consent_gate.md"], [0.8, 0.7])
    assert s.passed
    assert s.first_relevant_rank == 2
    assert s.reciprocal_rank == 0.5
    assert s.top_similarity == 0.8


def test_retrieval_miss_fails():
    s = score_retrieval(_VERISIM, ["README.md"], [0.6])
    assert not s.passed and s.reciprocal_rank == 0.0


def test_out_of_scope_retrieval_passes_only_when_cutoff_filters_everything():
    assert score_retrieval(_NONE, [], []).passed
    assert not score_retrieval(_NONE, ["README.md"], [0.5]).passed


@pytest.mark.parametrize(
    ("expected", "calls", "ok"),
    [
        ("verisim", ["verisim_agent"], True),
        ("verisim", ["verisim_agent", "ares_agent"], False),
        ("verisim", ["search_all_projects"], False),
        ("verisim", ["verisim_agent", "search_all_projects"], True),
        ("cross", ["search_all_projects"], True),
        ("cross", ["verisim_agent", "ares_agent"], True),
        ("cross", ["ares_agent"], False),
        ("none", [], None),
    ],
)
def test_routing_correct(expected, calls, ok):
    assert routing_correct(expected, calls) == ok


def test_e2e_pass_requires_routing_citation_and_keywords():
    good = score_e2e(
        _VERISIM, "Trust gates it (docs/consent_gate.md > State machine)", ["verisim_agent"], 3.0
    )
    assert good.passed

    no_cite = score_e2e(_VERISIM, "Trust gates it.", ["verisim_agent"], 3.0)
    assert not no_cite.passed and no_cite.cited_expected_source is False

    no_kw = score_e2e(_VERISIM, "See docs/consent_gate.md", ["verisim_agent"], 3.0)
    assert not no_kw.passed and no_kw.missing_keywords == ["trust"]


def test_e2e_out_of_scope_must_abstain():
    assert score_e2e(_NONE, "I don't have docs on that.", [], 1.0).passed
    cited = score_e2e(_NONE, "Per (README.md > Run) you should...", ["ares_agent"], 1.0)
    assert not cited.passed and cited.abstained is False


def test_degraded_answer_is_flagged_and_excluded_from_quality_rates():
    degraded = score_e2e(
        _VERISIM,
        "devdesk_router couldn't reach the model right now (ServerError); please try again shortly.",
        [],
        60.0,
    )
    good = score_e2e(_VERISIM, "trust (docs/consent_gate.md > X)", ["verisim_agent"], 4.0)
    assert degraded.degraded and not degraded.passed

    summary = summarize_e2e([degraded, good])
    assert summary["degraded"] == 1
    assert summary["pass_rate"] == 1.0
    assert summary["routing_accuracy"] == 1.0
    assert summary["latency_max_s"] == 4.0


def test_summarize_retrieval_splits_in_and_out_of_scope():
    scores = [
        score_retrieval(_VERISIM, ["docs/consent_gate.md"], [0.9]),
        score_retrieval(_VERISIM, ["a.md", "docs/consent_gate.md"], [0.9, 0.8]),
        score_retrieval(_NONE, ["a.md"], [0.4]),
    ]
    summary = summarize_retrieval(scores)
    assert summary["hit_at_k"] == 1.0
    assert summary["hit_at_1"] == 0.5
    assert summary["mrr"] == 0.75
    assert summary["out_of_scope_rejected"] == 0.0
