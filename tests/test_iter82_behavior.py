"""Black-box behavioral tests for iter-82: the month-name and amount-label
regexes in ``reflect_mcp.heuristics`` get first-letter lookaheads (and
prefix-factored month alternatives) with byte-identical behavior.

Covers the Expected Behaviors from pm.md:
  1. ``_MONTH_FIRST_RE`` / ``_DAY_FIRST_RE`` (span, groups) identity vs oracle.
  2. ``_LABELLED_AMOUNT_RES[role]`` (span, groups) identity vs oracle.
  3. Every label alternative derived from ``_ROLE_LABELS`` still matches.
  4. ``run_heuristics`` report identity over 400 seeded docs (differential).
  5. Relative speed: shipped ``_MONTH_FIRST_RE`` >= 2.0x the oracle.
  6. ``make eval`` headline lines unchanged, exit 0.
  7. ``validate --heuristics-only`` stdout on golden inv-001 unchanged vs the
     base commit (minus ``trace_id``).

Isolation contract: the "oracle" below is HEAD 26de4e7's regex construction
copied verbatim (as the spec requires) so it cannot drift with src. Every other
assertion drives public objects / the CLI and checks observable output only.
"""

from __future__ import annotations

import dataclasses
import importlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from reflect_mcp.schema import InvoiceData

REPO = Path(__file__).resolve().parents[1]
BASE_SHA = "26de4e7"


def _h():
    # Resolve at call time: other modules re-import the package.
    return importlib.import_module("reflect_mcp.heuristics")


# --------------------------------------------------------------------------
# Oracle: HEAD 26de4e7 construction, copied verbatim.
# --------------------------------------------------------------------------
_O_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_O_MONTHS = {name: i for i, name in enumerate(_O_MONTH_NAMES, 1)}
_O_MONTHS.update({name[:3]: i for i, name in enumerate(_O_MONTH_NAMES, 1)})
_MONTH_RE_ORACLE = "|".join(sorted(_O_MONTHS, key=len, reverse=True))

_O_MONTH_FIRST_RE = re.compile(
    rf"(?<![a-z])({_MONTH_RE_ORACLE})\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)", re.IGNORECASE
)
_O_DAY_FIRST_RE = re.compile(
    rf"(?<!\d)(\d{{1,2}})\s+({_MONTH_RE_ORACLE})\s+(\d{{4}})(?!\d)", re.IGNORECASE
)
_O_ROLE_LABELS = {
    "subtotal": r"sub-?total|sub total|net",
    "tax": r"tax|vat|gst",
    "total": r"(?<!sub)(?<!sub-)(?<!sub )(?:grand total|amount due|total)",
}
_O_LABELLED_AMOUNT_RES = {
    role: re.compile(
        rf"(?<![a-z])(?:{labels})(?![a-z])[^\d\n]{{0,25}}?(?:[$€£]\s*)?"
        r"((?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2})(?!\d)",
        re.IGNORECASE,
    )
    for role, labels in _O_ROLE_LABELS.items()
}

ROLES = ("subtotal", "tax", "total")


def _sg(p, text):
    return [(m.span(), m.groups()) for m in p.finditer(text)]


def _alt_case(s):
    return "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(s))


def _cases(s):
    return [s.lower(), s.title(), s.upper(), _alt_case(s)]


# --------------------------------------------------------------------------
# EB1: month regex identity
# --------------------------------------------------------------------------
def _month_corpus():
    spellings = list(_O_MONTH_NAMES) + [n[:3] for n in _O_MONTH_NAMES]
    lines = []
    for sp in spellings:
        for m in _cases(sp):
            lines += [
                f"{m} 5, 2026",
                f"{m} 05 2026",
                f"5 {m} 2026",
                f"x{m} 5, 2026",
                f"{m}s 5, 2026",
                f"{m}\n5, 2026",
            ]
    lines += ["Sept 3, 2026", "SEPT 3, 2026", "12 Sept 2026", "Date: Mayo 3, 2026"]
    return "\n".join(lines) + "\n"


class TestEB1MonthIdentity:
    def test_month_first_identity(self):
        corpus = _month_corpus()
        want = _sg(_O_MONTH_FIRST_RE, corpus)
        got = _sg(_h()._MONTH_FIRST_RE, corpus)
        assert want, "oracle produced no matches -- corpus broken"
        assert got == want

    def test_day_first_identity(self):
        corpus = _month_corpus()
        want = _sg(_O_DAY_FIRST_RE, corpus)
        got = _sg(_h()._DAY_FIRST_RE, corpus)
        assert want
        assert got == want

    def test_every_month_matched(self):
        corpus = _month_corpus()
        h = _h()
        seen = set()
        for _, groups in _sg(h._MONTH_FIRST_RE, corpus):
            seen.add(_O_MONTHS[groups[0].lower()])
        for _, groups in _sg(h._DAY_FIRST_RE, corpus):
            seen.add(_O_MONTHS[groups[1].lower()])
        assert seen == set(range(1, 13))


# --------------------------------------------------------------------------
# EB2: amount-label identity
# --------------------------------------------------------------------------
_LABEL_WORDS = ("subtotal", "sub-total", "sub total", "net", "tax", "vat", "gst",
                "grand total", "amount due", "total")


def _label_corpus():
    lines = []
    for i, w in enumerate(_LABEL_WORDS):
        for c in (w.lower(), w.title(), w.upper()):
            lines.append(f"{c}: {i + 10}.{i:02d}")
            lines.append(f"{c} ...... $ {i},{i:03d}.50")
    lines += [
        "Internet 9.99", "taxi 12.00", "Sub-total: 10.00", "Subtotal Total 5.00",
        "Total: € 9.99", "Total: $1,234.56", "Total: 15", "Nettotal 3.00",
        "Grand Total: £ 77.10", "Amount Due 1,000,000.01",
    ]
    return "\n".join(lines) + "\n"


class TestEB2LabelIdentity:
    @pytest.mark.parametrize("role", ROLES)
    def test_label_identity(self, role):
        corpus = _label_corpus()
        want = _sg(_O_LABELLED_AMOUNT_RES[role], corpus)
        got = _sg(_h()._LABELLED_AMOUNT_RES[role], corpus)
        assert want
        assert got == want

    def test_roles_unchanged(self):
        assert set(_h()._LABELLED_AMOUNT_RES) == set(ROLES)


# --------------------------------------------------------------------------
# EB3: no label can silently stop matching (derived at run time)
# --------------------------------------------------------------------------
def _split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _derive_alternatives(labels):
    s = labels
    while True:  # strip leading lookarounds (?<!..) (?<=..) (?=..) (?!..)
        m = re.match(r"\(\?(?:<!|<=|=|!)[^()]*\)", s)
        if not m:
            break
        s = s[m.end():]
    if s.startswith("(?:") and s.endswith(")"):
        s = s[3:-1]
    out = []
    for alt in _split_top(s):
        if "-?" in alt:
            out += [alt.replace("-?", "-"), alt.replace("-?", "")]
        else:
            out.append(alt)
    return out


class TestEB3EveryLabelMatches:
    def test_derivation_sees_known_labels(self):
        # Sanity: the derivation applied to HEAD's strings finds every label.
        derived = set()
        for role in ROLES:
            derived |= set(_derive_alternatives(_O_ROLE_LABELS[role]))
        assert derived == set(_LABEL_WORDS)

    @pytest.mark.parametrize("role", ROLES)
    def test_every_label_matches(self, role):
        h = _h()
        alts = _derive_alternatives(h._ROLE_LABELS[role])
        assert alts and all(re.fullmatch(r"[a-z \-]+", a) for a in alts), alts
        for label in alts:
            for form in (label, label.upper()):
                line = f"{form}: 12.34"
                ms = list(h._LABELLED_AMOUNT_RES[role].finditer(line))
                assert len(ms) == 1, (role, line, ms)
                assert ms[0].group(1) == "12.34", (role, line)


# --------------------------------------------------------------------------
# EB4: report identity (differential, 400 seeded docs)
# --------------------------------------------------------------------------
_VENDORS = ("ACME Corp", "Globex Ltd", "Initech", "Umbrella GmbH", "Stark Ind")
_DATE_LABELS = ("Date:", "Invoice date:", "Due date:", "Ship date:", "Shipped:",
                "Pay by", "Delivery date:", "Issued", "")


def _fmt_date(rng, y, mo, d):
    name = _O_MONTH_NAMES[mo - 1]
    k = rng.randrange(7)
    if k == 0:
        return f"{y:04d}-{mo:02d}-{d:02d}"
    if k == 1:
        return f"{mo:02d}/{d:02d}/{y}"
    if k == 2:
        return f"{d:02d}.{mo:02d}.{y}"
    if k == 3:
        return f"{rng.choice(_cases(name))} {d}, {y}"
    if k == 4:
        return f"{rng.choice(_cases(name[:3]))} {d:02d} {y}"
    if k == 5:
        return f"{d} {rng.choice(_cases(name))} {y}"
    return f"{d} {rng.choice(_cases(name[:3]))} {y}"


def _gen_doc(rng):
    lines = [rng.choice(_VENDORS), f"Invoice #INV-{rng.randrange(1000, 9999)}"]
    dates = []
    for _ in range(rng.randrange(1, 4)):
        y, mo, d = 2026, rng.randrange(1, 13), rng.randrange(1, 29)
        dates.append(f"{y:04d}-{mo:02d}-{d:02d}")
        lines.append(f"{rng.choice(_DATE_LABELS)} {_fmt_date(rng, y, mo, d)}".strip())
    for i in range(rng.randrange(0, 5)):
        lines.append(f"Widget {i} qty {rng.randrange(1, 9)} {rng.randrange(1, 500)}.{rng.randrange(100):02d}")
    sub = round(rng.randrange(100, 100000) / 100, 2)
    tax = round(sub * rng.choice((0.0, 0.05, 0.085, 0.2)), 2)
    tot = round(sub + tax, 2)
    sl = rng.choice(("Subtotal", "Sub-total", "SUB TOTAL", "Net", "Items"))
    tl = rng.choice(("Tax", "VAT", "GST", "tax", "Taxi"))
    gl = rng.choice(("Total", "Grand Total", "Amount due", "TOTAL", "Sub total"))
    cur = rng.choice(("$", "€", "£", ""))
    lines.append(f"{sl}: {cur}{sub:,.2f}")
    lines.append(f"{tl} ..... {tax:.2f}")
    lines.append(f"{gl}: {cur} {tot:.2f}")
    rng.shuffle(lines)
    text = "\n".join(lines)

    def maybe(v, wrong):
        r = rng.random()
        return v if r < 0.55 else (wrong if r < 0.9 else None)

    data = InvoiceData(
        vendor_name=lines[0] if rng.random() < 0.5 else rng.choice(_VENDORS),
        invoice_date=maybe(rng.choice(dates), f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"),
        subtotal=maybe(sub, tot),
        tax=maybe(tax, sub),
        total=maybe(tot, round(tot + 1, 2)),
        currency=rng.choice(("USD", "EUR", None)),
    )
    return text, data


def _docs():
    rng = random.Random(82)
    return [_gen_doc(rng) for _ in range(400)]


def _reports(h, docs):
    return [dataclasses.asdict(h.run_heuristics(t, d)) for t, d in docs]


def _patch_oracle(monkeypatch, h):
    monkeypatch.setattr(h, "_MONTH_FIRST_RE", _O_MONTH_FIRST_RE)
    monkeypatch.setattr(h, "_DAY_FIRST_RE", _O_DAY_FIRST_RE)
    monkeypatch.setattr(
        h, "_DATE_PATTERNS",
        (h._ISO_DATE_RE, h._NUMERIC_DATE_RE, _O_MONTH_FIRST_RE, _O_DAY_FIRST_RE),
    )
    monkeypatch.setattr(h, "_LABELLED_AMOUNT_RES", dict(_O_LABELLED_AMOUNT_RES))


class TestEB4ReportIdentity:
    def test_date_patterns_order(self):
        h = _h()
        assert h._DATE_PATTERNS == (
            h._ISO_DATE_RE, h._NUMERIC_DATE_RE, h._MONTH_FIRST_RE, h._DAY_FIRST_RE
        )

    def test_differential_400(self, monkeypatch):
        h = _h()
        docs = _docs()
        shipped = _reports(h, docs)
        _patch_oracle(monkeypatch, h)
        oracle = _reports(h, docs)
        role_keys = {"invoice_date", "subtotal", "tax", "total"}
        exercised = sum(1 for r in oracle if role_keys & set(r["reasons"]))
        assert exercised >= 50, exercised
        for i, (a, b) in enumerate(zip(shipped, oracle)):
            assert a == b, (i, docs[i][0])

    def test_patch_is_consumed(self, monkeypatch):
        # The patched names must actually drive run_heuristics, or EB4 is vacuous.
        h = _h()
        docs = _docs()[:120]
        base = _reports(h, docs)
        never = re.compile(r"(?!x)x(\d)")
        monkeypatch.setattr(h, "_LABELLED_AMOUNT_RES", {r: never for r in ROLES})
        monkeypatch.setattr(h, "_MONTH_FIRST_RE", never)
        monkeypatch.setattr(h, "_DAY_FIRST_RE", never)
        monkeypatch.setattr(h, "_DATE_PATTERNS", (h._ISO_DATE_RE, h._NUMERIC_DATE_RE, never, never))
        broken = _reports(h, docs)
        assert broken != base


# --------------------------------------------------------------------------
# EB5: relative speed (fails on HEAD where shipped == oracle)
# --------------------------------------------------------------------------
def _month_free_text():
    rng = random.Random(5)
    return "\n".join(
        f"Widget {i} qty {rng.randrange(1, 99)} shipped 2026-03-{rng.randrange(1, 29):02d} "
        f"price {rng.randrange(1, 9999)}.{rng.randrange(100):02d}"
        for i in range(4000)
    )


def _best(p, t, n=5):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        list(p.finditer(t))
        best = min(best, time.perf_counter() - t0)
    return best


class TestEB5Speed:
    def test_month_first_at_least_2x(self):
        t = _month_free_text()
        assert len(t.encode()) >= 190_000
        assert not re.search(_MONTH_RE_ORACLE, t, re.IGNORECASE)
        p = _h()._MONTH_FIRST_RE
        assert list(p.finditer(t)) == [] and list(_O_MONTH_FIRST_RE.finditer(t)) == []
        ratio = 0.0
        for _ in range(3):  # retry noisy runs; any one clean measurement suffices
            ratio = _best(_O_MONTH_FIRST_RE, t) / _best(p, t)
            if ratio >= 2.0:
                break
        assert ratio >= 2.0, f"shipped only {ratio:.2f}x faster than oracle"


# --------------------------------------------------------------------------
# EB6: eval unchanged
# --------------------------------------------------------------------------
class TestEB6Eval:
    def test_make_eval_headline(self):
        if shutil.which("make") is None:
            pytest.skip("make not available")
        r = subprocess.run(["make", "eval"], cwd=REPO, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "field accuracy    : 91.2%  (83/91)" in r.stdout  # iter 83: inv-011; iter 84: inv-012; iter 89: inv-013
        assert "silent-wrong      : 0" in r.stdout


# --------------------------------------------------------------------------
# EB7: CLI surface unchanged vs base commit
# --------------------------------------------------------------------------
def _golden(row_id):
    for line in (REPO / "eval" / "golden" / "dataset.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] == row_id:
            return row
    raise AssertionError(row_id)


def _validate(src_dir, row):
    env = dict(os.environ, PYTHONPATH=f"{src_dir}{os.pathsep}.")
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
         "--data", json.dumps(row["expected"])],
        input=row["text"], capture_output=True, text=True, cwd=REPO, env=env, timeout=120,
    )
    out = json.loads(r.stdout)
    out.pop("trace_id", None)
    return r.returncode, out


class TestEB7Surface:
    def test_validate_inv001_matches_base(self, tmp_path):
        r = subprocess.run(["git", "archive", BASE_SHA, "src"], cwd=REPO, capture_output=True)
        if r.returncode != 0:
            pytest.skip("base commit not available")
        with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
            tf.extractall(tmp_path)
        row = _golden("inv-001")
        base = _validate(tmp_path / "src", row)
        cur = _validate(REPO / "src", row)
        assert cur == base
        assert cur[1].get("status")
