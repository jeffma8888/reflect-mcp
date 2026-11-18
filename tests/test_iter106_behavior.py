"""Black-box behavioral tests for iter-106 (roadmap #30): a retry tells the
model why its previous reply was rejected.

On attempt N>=2, the user prompt is the attempt-1 user prompt plus
``"\\n\\nYour previous reply was rejected: " + str(exc)[:300] +
"\\nReply with only the corrected JSON object."``, where ``exc`` is the error of
the IMMEDIATELY preceding attempt. Attempt 1, system prompts, retry logs,
sleeps, error wrapping and offline surfaces stay as before.

Covers EB1-EB10 from pm.md. EB2-EB6 are the discriminators (red on the
pre-change tree, which re-sent a byte-identical prompt). EB1 and EB7-EB10 are
preservation checks and are green on both trees by design.

Isolation contract: only public interfaces are driven (Generator, Critic,
ParseError/parse_json_object, the schema models, the frozen eval helper),
with stubbed list-backed clients; no network, ``time.sleep`` patched.
"""

from __future__ import annotations

import json
import logging
import time

import pytest
from pydantic import ValidationError

from reflect_mcp import critic as critic_mod
from reflect_mcp import generator as generator_mod
from reflect_mcp.critic import Critic, CriticError
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.parsing import ParseError, parse_json_object
from reflect_mcp.schema import CritiqueResult, InvoiceData
from tests._golden import FROZEN_EVAL_SHA256, frozen_eval_sha256
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all

PHRASE = "\n\nYour previous reply was rejected: "
TAIL = "\nReply with only the corrected JSON object."

BAD = "not json at all"
BAD_LIST = "[1]"  # parses as JSON but is not an object: a distinct ParseError
VALID_INV = json.dumps(CORRECT_EXTRACTION)
VALID_CRIT = critique_all(0.95, 0.9, summary="all good")

FORMATTED_TOTAL = json.dumps({**CORRECT_EXTRACTION, "total": "$1,234.00"})
THREE_BAD_AMOUNTS = json.dumps(
    {**CORRECT_EXTRACTION, "subtotal": "abc", "tax": "def", "total": "ghi"}
)

GEN_USER = "Extract the invoice fields from this document:\n\n" + SOURCE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RecordingClient:
    """List-backed LLMClient that records every complete() call."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def stage(self, name: str) -> list[dict]:
        return [
            dict(getattr(r, "fields", {}) or {})
            for r in self.records
            if (getattr(r, "fields", {}) or {}).get("stage") == name
        ]


@pytest.fixture
def logcap():
    lg = logging.getLogger("reflect_mcp")
    h = _Capture()
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


@pytest.fixture
def sleeps(monkeypatch):
    rec: list = []
    monkeypatch.setattr(time, "sleep", lambda *a, **_k: rec.append(a[0] if a else None))
    return rec


def _extraction() -> InvoiceData:
    return InvoiceData.model_validate(CORRECT_EXTRACTION)


def _crit_user(text: str, extraction: InvoiceData) -> str:
    return (
        "SOURCE DOCUMENT:\n" + text
        + "\n\nCANDIDATE EXTRACTION (JSON):\n"
        + json.dumps(extraction.model_dump(), sort_keys=True)
        + "\n\nVerify each field against the source and return your critique JSON."
    )


def _parse_error_str(raw: str) -> str:
    with pytest.raises(ParseError) as ei:
        parse_json_object(raw)
    return str(ei.value)


def _validation_error_str(raw: str) -> str:
    with pytest.raises(ValidationError) as ei:
        InvoiceData.model_validate(parse_json_object(raw))
    return str(ei.value)


def _suffix(reason: str) -> str:
    return PHRASE + reason[:300] + TAIL


def _reason_segment(user: str, base: str) -> str:
    assert user.startswith(base + PHRASE), user[len(base):len(base) + 80]
    assert user.endswith(TAIL)
    return user[len(base + PHRASE):len(user) - len(TAIL)]


# ---------------------------------------------------------------------------
# EB1: attempt 1 unchanged
# ---------------------------------------------------------------------------

def test_eb1_first_attempt_prompt_unchanged(logcap, sleeps):
    client = _RecordingClient([VALID_INV])
    out = Generator(client, max_retries=1).generate(SOURCE, trace_id="e1")
    assert out == InvoiceData.model_validate(CORRECT_EXTRACTION)
    assert client.calls == [{"system": generator_mod.GENERATOR_SYSTEM, "user": GEN_USER}]
    assert PHRASE not in client.calls[0]["user"]
    assert logcap.stage("generate.retry") == []
    assert sleeps == []


# ---------------------------------------------------------------------------
# EB2: Generator retry carries the parse error
# ---------------------------------------------------------------------------

def test_eb2_generator_retry_carries_parse_error(logcap, sleeps):
    client = _RecordingClient([BAD, VALID_INV])
    out = Generator(client, max_retries=1).generate(SOURCE, trace_id="e2")
    assert out == InvoiceData.model_validate(CORRECT_EXTRACTION)
    assert len(client.calls) == 2
    assert client.calls[0]["user"] == GEN_USER
    reason = _parse_error_str(BAD)
    assert client.calls[1]["user"] == GEN_USER + _suffix(reason)
    # Same string as the retry log's error field, cut to 300.
    retries = logcap.stage("generate.retry")
    assert len(retries) == 1
    assert client.calls[1]["user"] == GEN_USER + _suffix(retries[0]["error"])
    for call in client.calls:
        assert call["system"] == generator_mod.GENERATOR_SYSTEM


# ---------------------------------------------------------------------------
# EB3: Generator retry carries a schema (ValidationError) reason
# ---------------------------------------------------------------------------

def test_eb3_generator_retry_carries_validation_error(logcap, sleeps):
    client = _RecordingClient([FORMATTED_TOTAL, VALID_INV])
    out = Generator(client, max_retries=1).generate(SOURCE, trace_id="e3")
    assert out == InvoiceData.model_validate(CORRECT_EXTRACTION)
    assert len(client.calls) == 2
    assert client.calls[0]["user"] == GEN_USER
    seg = _reason_segment(client.calls[1]["user"], GEN_USER)
    assert seg == _validation_error_str(FORMATTED_TOTAL)[:300]
    assert "total" in seg
    retries = logcap.stage("generate.retry")
    assert len(retries) == 1
    assert seg == retries[0]["error"][:300]


# ---------------------------------------------------------------------------
# EB4: Critic retry carries its own error only
# ---------------------------------------------------------------------------

def test_eb4_critic_retry_carries_own_error(logcap, sleeps):
    ext = _extraction()
    client = _RecordingClient([BAD, VALID_CRIT])
    out = Critic(client, max_retries=1).critique(SOURCE, ext, trace_id="e4")
    assert out == CritiqueResult.model_validate(json.loads(VALID_CRIT))
    assert len(client.calls) == 2
    base = _crit_user(SOURCE, ext)
    assert client.calls[0]["user"] == base
    assert client.calls[1]["user"] == base + _suffix(_parse_error_str(BAD))
    for call in client.calls:
        assert call["system"] == critic_mod.CRITIC_SYSTEM
    seg = _reason_segment(client.calls[1]["user"], base)
    assert seg == _parse_error_str(BAD)[:300]
    # Decorrelation: nothing from the Generator side leaks into the suffix.
    suffix = client.calls[1]["user"][len(base):]
    assert generator_mod.GENERATOR_SYSTEM not in suffix
    assert generator_mod.GENERATOR_SYSTEM[:40] not in suffix
    assert VALID_INV not in suffix


def test_eb4_critic_schema_error_reason(logcap, sleeps):
    bad_crit = json.dumps({"field_verdicts": "nope", "overall_confidence": 0.9})
    ext = _extraction()
    client = _RecordingClient([bad_crit, VALID_CRIT])
    Critic(client, max_retries=1).critique(SOURCE, ext, trace_id="e4b")
    base = _crit_user(SOURCE, ext)
    seg = _reason_segment(client.calls[1]["user"], base)
    retries = logcap.stage("critique.retry")
    assert len(retries) == 1
    assert seg == retries[0]["error"][:300]
    assert "field_verdicts" in seg


# ---------------------------------------------------------------------------
# EB5: reason truncated to 300 chars
# ---------------------------------------------------------------------------

def test_eb5_reason_truncated_to_300(logcap, sleeps):
    full = _validation_error_str(THREE_BAD_AMOUNTS)
    assert len(full) > 300  # guard: the input really exercises truncation
    client = _RecordingClient([THREE_BAD_AMOUNTS, VALID_INV])
    Generator(client, max_retries=1).generate(SOURCE, trace_id="e5")
    user2 = client.calls[1]["user"]
    seg = _reason_segment(user2, GEN_USER)
    assert seg == full[:300]
    assert len(seg) == 300
    assert len(user2) == len(GEN_USER) + len(PHRASE) + 300 + len(TAIL)
    # The log still carries the full, untruncated error.
    assert logcap.stage("generate.retry")[0]["error"] == full


def test_eb5_short_reason_not_padded(sleeps):
    # Counterpart: a short ParseError is carried whole (no padding / cut).
    reason = _parse_error_str(BAD)
    assert len(reason) < 300
    client = _RecordingClient([BAD, VALID_INV])
    Generator(client, max_retries=1).generate(SOURCE)
    assert _reason_segment(client.calls[1]["user"], GEN_USER) == reason


# ---------------------------------------------------------------------------
# EB6: latest reason only, never cumulative
# ---------------------------------------------------------------------------

def test_eb6_latest_reason_only(logcap, sleeps):
    err_a = _parse_error_str(BAD)
    err_b = _parse_error_str(BAD_LIST)
    assert err_a != err_b
    client = _RecordingClient([BAD, BAD_LIST, VALID_INV])
    out = Generator(client, max_retries=2).generate(SOURCE, trace_id="e6")
    assert out == InvoiceData.model_validate(CORRECT_EXTRACTION)
    assert len(client.calls) == 3
    assert client.calls[0]["user"] == GEN_USER
    assert client.calls[1]["user"] == GEN_USER + _suffix(err_a)
    assert client.calls[2]["user"] == GEN_USER + _suffix(err_b)
    assert err_a not in client.calls[2]["user"]
    assert client.calls[2]["user"].count(PHRASE.strip()) == 1
    assert sleeps == [0.5, 0.5]


def test_eb6_critic_latest_reason_only(sleeps):
    ext = _extraction()
    client = _RecordingClient([BAD, BAD_LIST, VALID_CRIT])
    Critic(client, max_retries=2).critique(SOURCE, ext)
    base = _crit_user(SOURCE, ext)
    assert client.calls[1]["user"] == base + _suffix(_parse_error_str(BAD))
    assert client.calls[2]["user"] == base + _suffix(_parse_error_str(BAD_LIST))


# ---------------------------------------------------------------------------
# EB7: exhaustion unchanged
# ---------------------------------------------------------------------------

def test_eb7_generator_exhaustion_unchanged(logcap, sleeps):
    client = _RecordingClient([BAD, BAD_LIST])
    with pytest.raises(GeneratorError) as ei:
        Generator(client, max_retries=1).generate(SOURCE, trace_id="e7")
    e = ei.value
    assert len(client.calls) == 2
    assert isinstance(e.__cause__, ParseError)
    assert str(e.__cause__) == _parse_error_str(BAD_LIST)  # the LAST error
    assert str(e) == "Generator produced invalid extraction: " + _parse_error_str(BAD_LIST)
    retries = logcap.stage("generate.retry")
    assert len(retries) == 1
    r = retries[0]
    assert r["trace_id"] == "e7"
    assert r["attempt"] == 1
    assert r["max_retries"] == 1
    assert r["error"] == _parse_error_str(BAD)
    assert sleeps == [0.5]


def test_eb7_generator_exhaustion_log_error_untruncated(logcap, sleeps):
    full = _validation_error_str(THREE_BAD_AMOUNTS)
    client = _RecordingClient([THREE_BAD_AMOUNTS, THREE_BAD_AMOUNTS])
    with pytest.raises(GeneratorError) as ei:
        Generator(client, max_retries=1).generate(SOURCE)
    assert isinstance(ei.value.__cause__, ValidationError)
    assert str(ei.value) == "Generator produced invalid extraction: " + full
    assert logcap.stage("generate.retry")[0]["error"] == full


def test_eb7_critic_exhaustion_unchanged(logcap, sleeps):
    client = _RecordingClient([BAD, BAD_LIST])
    with pytest.raises(CriticError) as ei:
        Critic(client, max_retries=1).critique(SOURCE, _extraction(), trace_id="e7c")
    e = ei.value
    assert len(client.calls) == 2
    assert isinstance(e.__cause__, ParseError)
    assert str(e) == "Critic produced invalid critique: " + _parse_error_str(BAD_LIST)
    retries = logcap.stage("critique.retry")
    assert len(retries) == 1
    assert retries[0]["attempt"] == 1
    assert retries[0]["max_retries"] == 1
    assert retries[0]["error"] == _parse_error_str(BAD)
    assert sleeps == [0.5]


# ---------------------------------------------------------------------------
# EB8: no retry when max_retries=0
# ---------------------------------------------------------------------------

def test_eb8_generator_no_retry(logcap, sleeps):
    client = _RecordingClient([BAD])
    with pytest.raises(GeneratorError) as ei:
        Generator(client, max_retries=0).generate(SOURCE)
    assert client.calls == [{"system": generator_mod.GENERATOR_SYSTEM, "user": GEN_USER}]
    assert str(ei.value) == "Generator produced invalid extraction: " + _parse_error_str(BAD)
    assert logcap.stage("generate.retry") == []
    assert sleeps == []


def test_eb8_critic_no_retry(sleeps):
    ext = _extraction()
    client = _RecordingClient([BAD])
    with pytest.raises(CriticError):
        Critic(client, max_retries=0).critique(SOURCE, ext)
    assert [c["user"] for c in client.calls] == [_crit_user(SOURCE, ext)]
    assert sleeps == []


# ---------------------------------------------------------------------------
# EB9: provider errors unchanged
# ---------------------------------------------------------------------------

def test_eb9_provider_error_propagates_unretried(logcap, sleeps):
    boom = RuntimeError("boom")
    client = _RecordingClient([boom, VALID_INV])
    with pytest.raises(RuntimeError) as ei:
        Generator(client, max_retries=1).generate(SOURCE)
    assert ei.value is boom
    assert len(client.calls) == 1
    assert logcap.stage("generate.retry") == []
    assert sleeps == []


def test_eb9_critic_provider_error_propagates_unretried(sleeps):
    boom = RuntimeError("boom")
    client = _RecordingClient([boom, VALID_CRIT])
    with pytest.raises(RuntimeError) as ei:
        Critic(client, max_retries=1).critique(SOURCE, _extraction())
    assert ei.value is boom
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# EB10: offline surfaces unchanged
# ---------------------------------------------------------------------------

def test_eb10_frozen_eval_identical():
    assert frozen_eval_sha256() == FROZEN_EVAL_SHA256
