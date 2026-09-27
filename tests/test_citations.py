import asyncio

from devdesk.citations import (
    UNVERIFIED_MARKER,
    record_hits,
    start_request,
    verify_answer,
)

RECORD = {
    ("README.md", "Ares > Requirements"),
    ("ARCHITECTURE.md", "Architecture > Game engine (`convex/engine`) > Engine state management"),
}


def test_retrieved_citation_is_kept():
    answer = "Needs Python 3.12 (README.md > Ares > Requirements)."
    assert verify_answer(answer, RECORD) == (answer, [])


def test_fabricated_citation_is_replaced_and_reported():
    answer = "Needs Python (devdesk_ares: docs/install_and_requirements.md > Installation & Requirements)"
    out, removed = verify_answer(answer, RECORD)
    assert out == f"Needs Python {UNVERIFIED_MARKER}"
    assert removed == ["docs/install_and_requirements.md > Installation & Requirements"]


def test_real_file_but_unretrieved_section_is_rejected():
    _, removed = verify_answer("x (README.md > Ares > Commands)", RECORD)
    assert removed == ["README.md > Ares > Commands"]


def test_project_prefix_and_shortened_header_are_accepted():
    for cited in ("(ares: README.md > Ares > Requirements)", "(README.md > Ares)"):
        assert verify_answer(f"x {cited}", RECORD)[1] == []


def test_headers_with_nested_parentheses():
    answer = "Loads state (ARCHITECTURE.md > Architecture > Game engine (`convex/engine`) > Engine state management)."
    assert verify_answer(answer, RECORD) == (answer, [])


def test_mixed_parenthesis_keeps_only_verified_parts():
    answer = "x (README.md > Ares > Requirements; fake.md > Nope)"
    out, removed = verify_answer(answer, RECORD)
    assert out == "x (README.md > Ares > Requirements)"
    assert removed == ["fake.md > Nope"]


def test_ordinary_parentheticals_are_untouched():
    answer = "Python 3.12 (or newer) is required (README.md > Ares > Requirements)."
    assert verify_answer(answer, RECORD) == (answer, [])


def test_record_is_per_request_context():
    async def request(name):
        record = start_request()
        record_hits([{"source_file": f"{name}.md", "header_path": "H"}])
        await asyncio.sleep(0)
        return record

    async def both():
        return await asyncio.gather(
            asyncio.create_task(request("a")), asyncio.create_task(request("b"))
        )

    a, b = asyncio.run(both())
    assert a == {("a.md", "H")} and b == {("b.md", "H")}


def test_record_hits_outside_a_request_is_a_noop():
    record_hits([{"source_file": "x.md", "header_path": "H"}])  # must not raise


# Formats seen in real eval answers (2026-09-27): none of these are fabricated.
REAL = {
    ("README.md", "Ares > What It Does"),
    ("README.md", "Ares > Architecture"),
    (".serena/memories/project/ares_overview.md", "Ares project overview"),
    ("docs/k.md", "Setup (note for log) > Caveats"),
    ("demo/V.md", "VeriSim — Architecture, Traceability and Validation Evidence"),
}


def test_comma_separated_citations_with_prefixes():
    answer = "x (devdesk_ares: README.md > Ares > What It Does, devdesk_ares: README.md > Ares > Architecture)"
    assert verify_answer(answer, REAL) == (answer, [])


def test_backticked_citations_with_label():
    answer = "x (Citations: `README.md > Ares > Architecture`, `.serena/memories/project/ares_overview.md > Ares project overview`)"
    assert verify_answer(answer, REAL) == (answer, [])


def test_uppercase_project_label_and_parenthesised_header():
    answer = "x (VERISIM: `docs/k.md > Setup (note for log) > Caveats`)"
    assert verify_answer(answer, REAL) == (answer, [])


def test_header_containing_a_comma_is_not_split():
    answer = "x (demo/V.md > VeriSim — Architecture, Traceability and Validation Evidence)"
    assert verify_answer(answer, REAL) == (answer, [])


def test_one_fabricated_among_comma_separated_real_ones():
    answer = "x (README.md > Ares > Architecture, fake.md > Made Up)"
    out, removed = verify_answer(answer, REAL)
    assert out == "x (README.md > Ares > Architecture)"
    assert removed == ["fake.md > Made Up"]
