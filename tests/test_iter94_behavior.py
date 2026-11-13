"""Black-box behavioral tests for iter-94 (#89): README check lists stay in sync
with the ``reflect_mcp.heuristics`` module docstring.

Covers the Expected Behaviors from pm.md:
  1. README section 2 (``**programmatic pass criterion**`` paragraph) names the
     omitted-total check with an explicit-null clause.
  2. README section 5 (``**Heuristics-only mode**`` paragraph) names it too.
  3. Section 5 says "date grounding" explicitly; the old "date validation and
     grounding" wording is gone.
  4. Section 7 (``- **Deterministic heuristics**`` bullet) still names every check.
  5. The docstring's ``  - label:`` bullets parse to exactly the 10 known labels.
  6. A {label: README phrase} map whose key set must EQUAL the parsed label set.
  7. Every phrase appears in every one of the three lists (case-insensitive,
     ``-`` folded to a space); each anchor matches exactly one README line.
  8. Non-vacuous: the pre-change section 2/5 wording (rebuilt in-process from
     the current lines, no VCS access) fails with exactly the missing pairs.
  9. (no src change / eval byte-identity are report-only, see tester.md.)

Isolation contract: this module reads README.md and the runtime ``__doc__`` of
the public heuristics module; it never reads implementation source. It spawns
no processes and reads no VCS revision, so it is safe in a committed clone or a
shallow checkout.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from reflect_mcp import heuristics

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

EXPECTED_LABELS = {
    "grounding",
    "arithmetic",
    "date format",
    "date grounding",
    "amount-role grounding",
    "party-role grounding",
    "date-role grounding",
    "identifier-role grounding",
    "currency-code shape",
    "omitted total",
}

LABEL_TO_PHRASE: dict[str, str] = {
    "grounding": "grounding",
    "arithmetic": "arithmetic reconciliation",
    "date format": "date valid",
    "date grounding": "date grounding",
    "amount-role grounding": "amount-role grounding",
    "party-role grounding": "party-role grounding",
    "date-role grounding": "date-role grounding",
    "identifier-role grounding": "identifier-role grounding",
    "currency-code shape": "currency-code shape",
    "omitted total": "omitted total",
}

# section name -> (anchor, mode); "substring" anchors may sit mid-line,
# "prefix" anchors must start the line.
ANCHORS: dict[str, tuple[str, str]] = {
    "§2": ("**programmatic pass criterion**", "substring"),
    "§5": ("**Heuristics-only mode**", "prefix"),
    "§7": ("- **Deterministic heuristics**", "prefix"),
}

_BULLET = re.compile(r"^  - ([^:\n]+):", re.MULTILINE)


# ---- helpers ---------------------------------------------------------------
def parse_labels(doc: str | None) -> set[str]:
    assert doc, "heuristics module has no docstring"
    return {m.group(1).strip() for m in _BULLET.finditer(doc)}


def norm(s: str) -> str:
    return s.replace("-", " ").lower()


def locate(text: str) -> dict[str, str]:
    """Return {section: line} for each anchor, asserting exactly one match."""
    lines = text.splitlines()
    found: dict[str, str] = {}
    for section, (anchor, mode) in ANCHORS.items():
        if mode == "prefix":
            hits = [ln for ln in lines if ln.startswith(anchor)]
        else:
            hits = [ln for ln in lines if anchor in ln]
        assert len(hits) == 1, f"{section} anchor {anchor!r} matched {len(hits)} lines"
        found[section] = hits[0]
    return found


def missing_pairs(text: str) -> list[tuple[str, str]]:
    sections = locate(text)
    return [
        (section, phrase)
        for section, line in sections.items()
        for phrase in LABEL_TO_PHRASE.values()
        if norm(phrase) not in norm(line)
    ]


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


# ---- EB1-EB3: section 2 / section 5 prose ------------------------------------
def test_eb1_section2_names_omitted_total_with_null_clause():
    line = locate(readme_text())["§2"]
    low = line.lower()
    assert "omitted total" in low or "omitted-total" in low
    assert "explicit null" in low, "§2 must say the check fires on an explicit null"
    assert "labels a total" in low
    assert "penaliz" in low


def test_eb2_section5_names_omitted_total_with_null_clause():
    line = locate(readme_text())["§5"]
    low = line.lower()
    assert "omitted total" in low or "omitted-total" in low
    assert "explicit null" in low, "§5 must say the check fires on an explicit null"
    assert "labels a total" in low
    assert "penaliz" in low


def test_eb3_section5_says_date_grounding_explicitly():
    line = locate(readme_text())["§5"]
    assert "date grounding" in line
    assert "date validation, date grounding" in line
    assert "date validation and grounding" not in line


# ---- EB4: section 7 still names every check ----------------------------------
def test_eb4_section7_bullet_names_every_phrase():
    line = locate(readme_text())["§7"]
    absent = [p for p in LABEL_TO_PHRASE.values() if norm(p) not in norm(line)]
    assert absent == [], f"§7 bullet lost check phrases: {absent}"


# ---- EB5: docstring parse ------------------------------------------------------
def test_eb5_docstring_parses_to_exactly_ten_labels():
    labels = parse_labels(heuristics.__doc__)
    assert labels == EXPECTED_LABELS
    assert len(labels) == 10


def test_eb5_parser_takes_text_up_to_first_colon_only():
    doc = "x\n\n  - foo bar: baz: qux\n  - date-role grounding: a\n     - nested: no\n"
    assert parse_labels(doc) == {"foo bar", "date-role grounding"}


# ---- EB6: map key set equals parsed label set ----------------------------------
def test_eb6_map_keys_equal_parsed_labels():
    labels = parse_labels(heuristics.__doc__)
    assert set(LABEL_TO_PHRASE) == labels, (
        f"docstring-only labels: {sorted(labels - set(LABEL_TO_PHRASE))}; "
        f"stale map keys: {sorted(set(LABEL_TO_PHRASE) - labels)}"
    )


def test_eb6_new_docstring_bullet_without_phrase_is_detected(monkeypatch):
    doc = heuristics.__doc__ + "\n  - brand new check: something new\n"
    monkeypatch.setattr(heuristics, "__doc__", doc)
    labels = parse_labels(heuristics.__doc__)
    assert set(LABEL_TO_PHRASE) != labels
    assert labels - set(LABEL_TO_PHRASE) == {"brand new check"}


def test_eb6_stale_map_key_is_detected(monkeypatch):
    doc = heuristics.__doc__.replace("  - omitted total:", "    omitted total;")
    assert doc != heuristics.__doc__
    monkeypatch.setattr(heuristics, "__doc__", doc)
    labels = parse_labels(heuristics.__doc__)
    assert set(LABEL_TO_PHRASE) - labels == {"omitted total"}


# ---- EB7: every phrase in every list --------------------------------------------
def test_eb7_each_anchor_matches_exactly_one_line():
    sections = locate(readme_text())
    assert set(sections) == {"§2", "§5", "§7"}
    assert len(set(sections.values())) == 3


def test_eb7_all_thirty_pairs_hold():
    sections = locate(readme_text())
    pairs = [(s, p) for s in sections for p in LABEL_TO_PHRASE.values()]
    assert len(pairs) == 30
    assert missing_pairs(readme_text()) == [], (
        f"README check lists miss (section, phrase): {missing_pairs(readme_text())}"
    )


def test_eb7_matching_is_case_insensitive_and_hyphen_folded():
    fixture = (
        "intro **programmatic pass criterion**: Grounding, AMOUNT ROLE grounding, "
        "party role grounding, identifier-role grounding, currency code shape, "
        "Arithmetic Reconciliation, date validity, date-grounding, date role "
        "grounding and the omitted-total check.\n"
        "**Heuristics-only mode** grounding amount-role grounding party-role "
        "grounding identifier-role grounding currency-code shape arithmetic "
        "reconciliation date validation date grounding date-role grounding omitted total\n"
        "- **Deterministic heuristics** grounding amount-role grounding party-role "
        "grounding identifier-role grounding currency-code shape arithmetic "
        "reconciliation ISO-date validation date grounding date-role grounding "
        "the omitted-total check\n"
    )
    assert missing_pairs(fixture) == []


def test_eb7_duplicate_or_absent_anchor_fails():
    text = readme_text()
    line5 = locate(text)["§5"]
    with pytest.raises(AssertionError, match="matched 2 lines"):
        locate(text + "\n" + line5 + "\n")
    with pytest.raises(AssertionError, match="matched 0 lines"):
        locate(text.replace("**programmatic pass criterion**", "**pass criterion**"))


# ---- EB8: non-vacuous counterfactual (pre-change wording, rebuilt in-process) ----
# Exact clause swaps that turn the shipped §2/§5 lines back into the pre-change
# wording. Each `new` must be present, so this counterfactual cannot silently
# become a no-op if the README is edited later.
S2_SWAP = (
    ", date-role grounding (an invoice date must not appear only as a due or ship "
    "date) and the omitted-total check (an explicit null `total` on a source that "
    "labels a total is penalized) are checkable",
    " and date-role grounding (an invoice date must not appear only as a due or "
    "ship date) are checkable",
)
S5_SWAP = (
    "date validation, date grounding, date-role grounding, omitted total (an "
    "explicit null `total` on a source that labels a total → penalized)).",
    "date validation and grounding, date-role grounding).",
)
S5_DATE_ONLY_SWAP = (
    "date validation and grounding, date-role grounding).",
    "date validation, date grounding, date-role grounding).",
)


def _pre_change_readme() -> str:
    text = readme_text()
    sections = locate(text)
    old2 = sections["§2"].replace(S2_SWAP[0], S2_SWAP[1])
    old5 = sections["§5"].replace(S5_SWAP[0], S5_SWAP[1])
    assert old2 != sections["§2"], "§2 swap clause not found in README"
    assert old5 != sections["§5"], "§5 swap clause not found in README"
    return text.replace(sections["§2"], old2).replace(sections["§5"], old5)


def test_eb8_pre_change_wording_fails_with_exact_pairs():
    assert missing_pairs(_pre_change_readme()) == [
        ("§2", "omitted total"),
        ("§5", "date grounding"),
        ("§5", "omitted total"),
    ]


def test_eb8_eb3_rewording_alone_clears_only_date_grounding():
    old = _pre_change_readme()
    old5 = locate(old)["§5"]
    fixed5 = old5.replace(S5_DATE_ONLY_SWAP[0], S5_DATE_ONLY_SWAP[1])
    assert fixed5 != old5
    assert missing_pairs(old.replace(old5, fixed5)) == [
        ("§2", "omitted total"),
        ("§5", "omitted total"),
    ]


def test_eb8_failure_message_names_missing_pairs():
    pairs = missing_pairs(_pre_change_readme())
    msg = f"README check lists miss (section, phrase): {pairs}"
    assert "('§2', 'omitted total')" in msg
    assert "('§5', 'omitted total')" in msg


# ---- EB8 guard: no subprocess, no VCS, no floating rev literals -----------------
def test_eb8_module_has_no_subprocess_or_vcs_literals():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    banned_mods = {"sub" + "process", "o" + "s"}
    banned_tokens = ("HE" + "AD", "g" + "it ", "." + "g" + "it")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned_mods, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned_mods, node.module
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for tok in banned_tokens:
                assert tok not in node.value, f"forbidden token in literal: {tok!r}"
    assert "/" + "Users/" not in Path(__file__).read_text(encoding="utf-8")
