"""Black-box behavioral tests for iter-93 (#85): ``run_heuristics`` stops
re-scanning the document, and its report stays byte-identical for every input.

Covers the Expected Behaviors from pm.md:
  1. Report identity over an in-process seeded differential corpus
     (13 golden docs x 4 text variants x >=10 field mutations), pinned to a
     frozen sha measured on base tree 1462fe6.
  2. Per-golden-row report digests (expected data and replay generator_raw),
     frozen from base 1462fe6.
  3. Offline eval stdout identical to base 1462fe6 (in-process, rc 0).
  4. ``_find_number_span(value, text)`` keeps its signature and early return.
  5. The full-document whitespace fold happens only when needed, at most once.
  6. The earliest date wins across written forms.
  7. Year-prefilter guard: every date pattern captures a literal 4-digit year.
  8. Wrong-role dates are still penalized behind year-free padding lines.

Isolation contract: every digest / literal below was measured by running this
module's own builders against an archive of base 1462fe6 (PYTHONPATH pointing
at the archive's ``src``), never by reading the implementation. The module
spawns no processes and reads no VCS revision, so it is safe in a committed
clone or a shallow checkout. EB9 (speed) is report-only and lives in the
tester report, not here.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import io
import json
import random
import sys
from pathlib import Path

import pytest

from reflect_mcp import heuristics as h
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.schema import InvoiceData

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"
GOLDEN_IDS = tuple(f"inv-{i:03d}" for i in range(1, 14))

# ---- frozen literals, measured on base 1462fe6 ------------------------------
BASE_CORPUS_ROWS = 1496
BASE_CORPUS_SHA = "a42365f224d7ca6d83eefc57542ab0f4b7da9765ed0647b32df188fafdd75278"
BASE_GOLDEN_DIGESTS: dict[str, dict[str, str]] = {
    "inv-001": {
        "expected": "9d219a5bb700e73c62909e3f4d85de6147fd21b91f49502a7de16702520bf33a",
        "replay": "9d219a5bb700e73c62909e3f4d85de6147fd21b91f49502a7de16702520bf33a",
    },
    "inv-002": {
        "expected": "fab809554262a00ff7d4311f59db8e2375a1cd7063f62099872e8df2834764a3",
        "replay": "fab809554262a00ff7d4311f59db8e2375a1cd7063f62099872e8df2834764a3",
    },
    "inv-003": {
        "expected": "89f18bb8bd877be4618db15ef7ce8176761a5d554d6d26fe71f9e5c3a3ff7d74",
        "replay": "89f18bb8bd877be4618db15ef7ce8176761a5d554d6d26fe71f9e5c3a3ff7d74",
    },
    "inv-004": {
        "expected": "3cd1fd606ff872fe337ff9062d4ca1e6649136485deb5899a2b6bf181010052d",
        "replay": "488cfeb2bedee8facb5ce8a9fb3bae072c6bbdd7f3b1568539bb0d3d27f9fdb7",
    },
    "inv-005": {
        "expected": "026e365525f5d893929b3210e75891db67d1c21eef328bb6a9f948076517b051",
        "replay": "026e365525f5d893929b3210e75891db67d1c21eef328bb6a9f948076517b051",
    },
    "inv-006": {
        "expected": "860a861b0aa05f70885bf474a0731fb4a9688eaefce891639c87ce7630e41b74",
        "replay": "2bcc73799706ca199b99df2c86351eefc3928e62fb00ebf9edc616cd30a8c289",
    },
    "inv-007": {
        "expected": "3dc290411d58e9489e113fa941ff2c854ecea4c2e003231dcd07f0ef9545839d",
        "replay": "cbedf000aa3afa8cd43970d7944f58c3d321907d89e2579cce6bfa47e63b88cb",
    },
    "inv-008": {
        "expected": "3a5c3fa97debcba6ca0bcef08d7a9c0eeb0348af616d5e04443ff1e4d89d326f",
        "replay": "aff6f5b41dd701d322b982665f9a5cfae51ba0498affdb439f8458081f7371f2",
    },
    "inv-009": {
        "expected": "ae97751cfbe8a003315cd789c1669f8b5817b741616017b96eee1f972f9b89e6",
        "replay": "80adc86a3c9f04ab4aaad95d8d1ffdb45227fd30533b19c94c7a1539a0265835",
    },
    "inv-010": {
        "expected": "84c3df23e1990f8f1ff86f8cae8c468cdacef60b795a9d296e9faf1e56bf2558",
        "replay": "ba64760c8ac4c55e145286ecbf7624bbf8b4af284311030b1a7763db60a0bd8a",
    },
    "inv-011": {
        "expected": "4c0cc0279dc780bb3d5b998377301e8f9acf4c02c4543207171542bb84edbf39",
        "replay": "cca680d32af19e3aed83334901ebc0cea0323a5f6c74b13068b55c9c2b0ae538",
    },
    "inv-012": {
        "expected": "62b3ad8abddf35071935abfebaa09c38f743015b69002401f7592837fa22801b",
        "replay": "62b3ad8abddf35071935abfebaa09c38f743015b69002401f7592837fa22801b",
    },
    "inv-013": {
        "expected": "1d3e06d683caf921366ed19b671c9cd63e1b3c9c85163ae637a45e860055747e",
        "replay": "ff07c0d443f3f226f792ca1f40a0fa84c46743137760008088cd0998a2870213",
    },
}
BASE_EVAL_INPUT_SHA = "3a8aa7f677d8da49aee1b7aad69c2927152c935cee9cfd3cda13bb8b4c0bb28b"
BASE_EVAL_STDOUT_SHA = "51d64f9d990954b6be47a37c4bfb0cd381905fabcb5e84d53af486a9cc446821"
BASE_WRONG_ROLE_REASON = (
    "date '2024-03-01' appears only as a due/ship/delivery date, not the invoice date"
)
# -----------------------------------------------------------------------------


def _canon(report) -> str:
    return json.dumps(
        {
            "f": report.factors,
            "r": report.reasons,
            "p": {k: [[s.start, s.end] for s in v] for k, v in report.provenance.items()},
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _golden_rows() -> dict[str, dict]:
    rows = {}
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    missing = [i for i in GOLDEN_IDS if i not in rows]
    assert not missing, f"frozen golden rows deleted: {missing}"
    return {i: rows[i] for i in GOLDEN_IDS}


def _replay() -> dict:
    return json.loads(REPLAY.read_text(encoding="utf-8"))


def _pad_lines() -> str:
    return "\n".join(
        f"2026-{i % 12 + 1:02d}-{i % 28 + 1:02d} item {i} qty {i % 5 + 1} @ {i % 37}.{i % 100:02d}"
        for i in range(50)
    )


def _variants(text: str) -> list[str]:
    return [
        text,
        text + "\n" + _pad_lines(),
        text.upper(),
        text.replace("\n", " \n"),
    ]


def _mutations(e: dict, rng: random.Random) -> list[dict]:
    out = [dict(e)]
    for f in ("subtotal", "tax", "total"):
        v = e.get(f)
        if v is not None:
            out += [
                dict(e, **{f: round(v + 0.01, 2)}),
                dict(e, **{f: round(v - 0.01, 2)}),
                dict(e, **{f: v * 10}),
                dict(e, **{f: None}),
            ]
    # decoy line-item amounts (present only in the padded variant)
    out += [dict(e, total=7.07), dict(e, subtotal=13.13)]
    # decoy dates: a padding-line date, a random other date, and null
    out += [
        dict(e, invoice_date="2026-08-08"),
        dict(e, invoice_date=f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"),
        dict(e, invoice_date=None),
    ]
    for f in ("vendor_name", "invoice_number", "currency"):
        s = e.get(f)
        if s:
            out += [
                dict(e, **{f: s[:-1] or "x"}),
                dict(e, **{f: s.lower()}),
                dict(e, **{f: s.replace(" ", "  ")}),
                dict(e, **{f: " ".join(s.split())}),
            ]
    return out


def _corpus_digest() -> tuple[int, str]:
    rng = random.Random(93)
    h_all = hashlib.sha256()
    n = 0
    for gid, row in _golden_rows().items():
        datas = _mutations(row["expected"], rng)
        for text in _variants(row["text"]):
            for d in datas:
                try:
                    data = InvoiceData(**d)
                except Exception:
                    continue
                h_all.update(_canon(run_heuristics(text, data)).encode("utf-8"))
                h_all.update(b"\n")
                n += 1
    return n, h_all.hexdigest()


def _golden_digests() -> dict[str, dict[str, str]]:
    replay = _replay()
    out: dict[str, dict[str, str]] = {}
    for gid, row in _golden_rows().items():
        entry = {"expected": _sha(_canon(run_heuristics(row["text"], InvoiceData(**row["expected"]))))}
        raw = replay[gid]["generator_raw"]
        try:
            data = InvoiceData(**json.loads(raw))
        except Exception:
            entry["replay"] = "unparseable"
        else:
            entry["replay"] = _sha(_canon(run_heuristics(row["text"], data)))
        out[gid] = entry
    return out


def _eval_input_sha() -> str:
    return hashlib.sha256(GOLDEN.read_bytes() + b"\0" + REPLAY.read_bytes()).hexdigest()


def _eval_stdout() -> tuple[int, str]:
    sys.path.insert(0, str(REPO / "eval"))
    try:
        import run_eval  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["run_eval.py"]
    try:
        with contextlib.redirect_stdout(buf):
            rc = run_eval.main()
    finally:
        sys.argv = argv
    return rc, buf.getvalue()


# ---- EB1 ---------------------------------------------------------------------
def test_eb1_differential_corpus_matches_base_digest():
    n, digest = _corpus_digest()
    assert n == BASE_CORPUS_ROWS
    assert n >= 13 * 3 * 10
    assert digest == BASE_CORPUS_SHA


# ---- EB2 ---------------------------------------------------------------------
def test_eb2_golden_row_reports_match_base():
    got = _golden_digests()
    assert set(got) == set(GOLDEN_IDS)
    diff = {k: (got[k], BASE_GOLDEN_DIGESTS.get(k)) for k in GOLDEN_IDS if got[k] != BASE_GOLDEN_DIGESTS.get(k)}
    assert not diff, f"golden report drift vs base: {diff}"


# ---- EB3 ---------------------------------------------------------------------
def test_eb3_offline_eval_stdout_matches_base():
    if _eval_input_sha() != BASE_EVAL_INPUT_SHA:
        pytest.skip("golden/replay inputs changed after iter 93; the base stdout pin no longer applies")
    rc, out = _eval_stdout()
    assert rc == 0
    assert "silent-wrong      : 0" in out
    assert _sha(out) == BASE_EVAL_STDOUT_SHA


# ---- EB4 ---------------------------------------------------------------------
def test_eb4_find_number_span_signature_and_first_match():
    assert list(inspect.signature(h._find_number_span).parameters) == ["value", "text"]
    text = "Qty 8 ... Tax 8.00"
    span = h._find_number_span(8.0, text)
    assert (span.start, span.end) == (4, 5)
    assert text[span.start:span.end] == "8"
    later = "Tax 8.00 and Qty 8"
    span2 = h._find_number_span(8.0, later)
    assert later[span2.start:span2.end] == "8.00"
    assert span2.start == 4
    assert h._find_number_span(9.0, text) is None


# ---- EB5 ---------------------------------------------------------------------
def _fold_counts(monkeypatch, text: str, data: InvoiceData):
    real = h._normalize
    calls: list[int] = []

    def rec(s):
        calls.append(len(s))
        return real(s)

    monkeypatch.setattr(h, "_normalize", rec)
    report = run_heuristics(text, data)
    monkeypatch.setattr(h, "_normalize", real)
    full = sum(1 for n in calls if n >= len(text) // 2)
    return report, full, len(calls)


_FILLER = "\n".join(f"widget {i} qty 2 @ 3.{i % 100:02d}" for i in range(300))


def test_eb5_no_full_fold_when_all_strings_verbatim(monkeypatch):
    text = "Acme Corp\nInvoice INV-1\nDate: 2024-03-01\n" + _FILLER + "\nTotal: 5.00 USD\n"
    data = InvoiceData(vendor_name="Acme Corp", invoice_number="INV-1", invoice_date="2024-03-01",
                       total=5.0, currency="USD")
    report, full, _ = _fold_counts(monkeypatch, text, data)
    assert full == 0
    assert report.factors == {}


@pytest.mark.parametrize(
    "src_vendor,src_number,data_vendor,data_number",
    [
        ("Acme  Corp", "INV-1", "Acme Corp", "INV-1"),
        ("Acme  Corp", "INV  77", "Acme Corp", "INV 77"),
        ("Acme\n Corp", "INV \t77", "Acme Corp", "INV 77"),
    ],
)
def test_eb5_fold_needed_happens_at_most_once(monkeypatch, src_vendor, src_number, data_vendor, data_number):
    text = f"{src_vendor}\nInvoice {src_number}\nDate: 2024-03-01\n" + _FILLER + "\nTotal: 5.00 USD\n"
    data = InvoiceData(vendor_name=data_vendor, invoice_number=data_number, invoice_date="2024-03-01",
                       total=5.0, currency="USD")
    report, full, total_calls = _fold_counts(monkeypatch, text, data)
    assert total_calls >= 1, "recorder never intercepted _normalize; EB5 checks would be vacuous"
    assert full <= 1
    assert "vendor_name" not in report.factors
    assert "invoice_number" not in report.factors


# ---- EB6 ---------------------------------------------------------------------
_FORMS = {
    "iso": "2024-03-01",
    "month_first": "March 1, 2024",
    "numeric": "03/01/2024",
    "day_first": "1 March 2024",
}


@pytest.mark.parametrize("first,second", [(a, b) for a in _FORMS for b in _FORMS if a != b])
def test_eb6_earliest_date_occurrence_wins(first, second):
    text = (
        "Acme Corp\nInvoice INV-1\n"
        f"Issued {_FORMS[first]}\n"
        + _FILLER
        + f"\nRef {_FORMS[second]}\nTotal: 5.00\n"
    )
    data = InvoiceData(vendor_name="Acme Corp", invoice_number="INV-1", invoice_date="2024-03-01", total=5.0)
    report = run_heuristics(text, data)
    spans = report.provenance["invoice_date"]
    start = text.index(_FORMS[first])
    assert (spans[0].start, spans[0].end) == (start, start + len(_FORMS[first]))
    assert "invoice_date" not in report.factors


def test_eb6_spec_example_both_orders():
    a = "Invoice date March 1, 2024\nnotes\nISO 2024-03-01\nTotal 5.00"
    b = "Invoice date 2024-03-01\nnotes\nwritten March 1, 2024\nTotal 5.00"
    data = InvoiceData(invoice_date="2024-03-01", total=5.0)
    sa = run_heuristics(a, data).provenance["invoice_date"][0]
    sb = run_heuristics(b, data).provenance["invoice_date"][0]
    assert a[sa.start:sa.end] == "March 1, 2024"
    assert b[sb.start:sb.end] == "2024-03-01"


# ---- EB7 ---------------------------------------------------------------------
def test_eb7_every_date_pattern_captures_a_four_digit_year():
    assert len(h._DATE_PATTERNS) == 4
    for p in h._DATE_PATTERNS:
        assert r"(\d{4})" in p.pattern, p.pattern
        assert p.search("03/14/26") is None
        assert p.search("March 1, 26") is None
        assert p.search("14 March 26") is None


def test_eb7_golden_date_tokens_contain_their_year():
    seen = 0
    per_pattern = [0] * len(h._DATE_PATTERNS)
    for row in _golden_rows().values():
        for i, p in enumerate(h._DATE_PATTERNS):
            for m in p.finditer(row["text"]):
                readings = h._date_readings(m)
                for y, _mo, _d in readings:
                    assert str(y) in m.group(0), (row["id"], m.group(0), y)
                    seen += 1
                per_pattern[i] += 1
    assert seen >= 13
    assert sum(1 for c in per_pattern if c) >= 2, per_pattern


# ---- EB8 ---------------------------------------------------------------------
_NO_YEAR_PAD = "\n".join(f"line {i} widget qty {i % 9 + 1} @ {i % 50}.50" for i in range(200))


@pytest.mark.parametrize("due_first", [True, False])
def test_eb8_due_only_date_still_penalized(due_first):
    due = "Due date: 2024-03-01"
    body = [due, _NO_YEAR_PAD] if due_first else [_NO_YEAR_PAD, due]
    text = "Acme Corp\nInvoice INV-1\n" + "\n".join(body) + "\nTotal: 5.00\n"
    data = InvoiceData(vendor_name="Acme Corp", invoice_number="INV-1", invoice_date="2024-03-01", total=5.0)
    report = run_heuristics(text, data)
    assert report.factors.get("invoice_date") == 0.3
    assert report.reasons["invoice_date"] == [BASE_WRONG_ROLE_REASON]


def test_eb8_neutral_invoice_date_line_clears_penalty():
    text = ("Acme Corp\nInvoice INV-1\nDue date: 2024-03-01\n" + _NO_YEAR_PAD
            + "\nInvoice date: 2024-03-01\nTotal: 5.00\n")
    data = InvoiceData(vendor_name="Acme Corp", invoice_number="INV-1", invoice_date="2024-03-01", total=5.0)
    report = run_heuristics(text, data)
    assert "invoice_date" not in report.factors
    assert "invoice_date" not in report.reasons


# ---- AC: committed-clone safety ------------------------------------------------
def test_module_spawns_no_process_and_reads_no_revision():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    banned_mods = {"subprocess", "pty", "multiprocessing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not {a.name.split(".")[0] for a in node.names} & banned_mods
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned_mods
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value != "HE" + "AD"
            assert not node.value.lstrip().startswith("gi" + "t ")
