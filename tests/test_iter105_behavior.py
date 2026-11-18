"""Black-box behavioral tests for iter-105: ``parse_json_object`` walks later
top-level balanced ``{...}`` segments when the first balanced one is not
valid JSON, so a stray ``{field: value}`` preamble no longer costs a retry.

Covers EB1-EB11 from pm.md. Only public interfaces are driven:
``reflect_mcp.parsing.parse_json_object`` / ``ParseError`` and the
``Generator`` / ``Critic`` classes over ``ScriptedLLMClient`` (offline).

EB8 note (ship hazard H2): the spec's "under 1 second" is NOT asserted as a
wall-clock bound. Each adversarial parse runs in a daemon thread joined with
a generous hang guard; the assertion is "terminates and raises ParseError",
which is what catches the quadratic / infinite-loop traps.
"""

from __future__ import annotations

import threading
import time

import pytest

from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.parsing import ParseError, parse_json_object
from reflect_mcp.schema import CritiqueResult, InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

EXPECTED = {"vendor_name": "Acme", "total": 10.0}

EB1 = (
    'I used the {field: value} format you asked for:\n'
    '{"vendor_name": "Acme", "total": 10.0}'
)
EB2 = (
    "Output schema {vendor_name, total} filled in below.\n\n"
    '{"vendor_name": "Acme", "total": 10.0}'
)
EB3 = (
    "Returning {vendor_name, total} and the rest:\n```json\n"
    '{"vendor_name": "Acme", "total": 10.0}\n```'
)
EB5 = '{field: value}\n{"vendor_name": "A", "meta": {"x": 1}, "total": 1'
CRIT_JSON = (
    '{"field_verdicts": [{"field": "total", "confidence": 0.9}], '
    '"overall_confidence": 0.9}'
)
EB10 = "Grading each {field: confidence} pair:\n" + CRIT_JSON
EB11_BAD = '{field: value}\n{"vendor_name": "A", "total": 1'

HANG_GUARD_S = 30.0


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return sleeps


# ---- EB1-EB3: stray brace preamble before the real object -------------------


@pytest.mark.parametrize("raw", [EB1, EB2, EB3], ids=["eb1", "eb2", "eb3_fenced"])
def test_eb1_3_stray_preamble_parses(raw: str) -> None:
    assert parse_json_object(raw) == EXPECTED


def test_eb1_counterpart_preamble_alone_still_fails() -> None:
    # Without the real object the stray preamble is still rejected.
    with pytest.raises(ParseError):
        parse_json_object("I used the {field: value} format you asked for:")


# ---- EB4: nested object returned whole --------------------------------------


def test_eb4_nested_real_object_returned_whole() -> None:
    out = parse_json_object('{a} {b} {"q": {"n": 1}}')
    assert out == {"q": {"n": 1}}
    assert out != {"n": 1}


# ---- EB5: no nested substitution on a truncated reply -----------------------


def test_eb5_truncated_outer_never_yields_nested_fragment() -> None:
    with pytest.raises(ParseError):
        parse_json_object(EB5)


def test_eb5_counterpart_closed_outer_parses_whole() -> None:
    # One-edit counterpart: close the outer object and it parses whole.
    assert parse_json_object(EB5 + "}") == {
        "vendor_name": "A",
        "meta": {"x": 1},
        "total": 1,
    }


# ---- EB6: all later segments malformed -> unchanged message prefix ----------


def test_eb6_all_malformed_raises_with_head_prefix() -> None:
    with pytest.raises(ParseError) as exc:
        parse_json_object("{x} then {y}")
    assert str(exc.value).startswith("Malformed JSON in model output")


# ---- EB7: HEAD-handled inputs unchanged -------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Here you go: {"a": 1} hope that helps', {"a": 1}),
        ('{"a": 1} and {note}', {"a": 1}),
        ('Shape: {"field": "value"}\n{"a": 1}', {"field": "value"}),
    ],
    ids=["bare", "fenced", "prose_wrapped", "first_valid_wins", "valid_example_first"],
)
def test_eb7_head_inputs_unchanged(raw: str, expected: dict) -> None:
    assert parse_json_object(raw) == expected


@pytest.mark.parametrize(
    ("raw", "prefix"),
    [
        ("no braces here", "No JSON object found"),
        ("[1, 2]", "Expected a JSON object"),
    ],
    ids=["no_braces", "array"],
)
def test_eb7_head_errors_unchanged(raw: str, prefix: str) -> None:
    with pytest.raises(ParseError) as exc:
        parse_json_object(raw)
    assert str(exc.value).startswith(prefix)


# ---- EB8: adversarial inputs terminate (hang guard, not a timing ratio) -----


@pytest.mark.parametrize(
    "raw",
    ["{x} " * 2048, "{x}" + "{" * 680, "{x" * 50000],
    ids=["repeated_stray", "unbalanced_tail", "never_closed"],
)
def test_eb8_adversarial_inputs_terminate_with_parse_error(raw: str) -> None:
    result: dict[str, object] = {}

    def run() -> None:
        try:
            result["value"] = parse_json_object(raw)
        except BaseException as e:  # noqa: BLE001 - recorded and asserted below
            result["exc"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(HANG_GUARD_S)
    assert not t.is_alive(), "parse_json_object did not terminate (hang/quadratic trap)"
    assert "value" not in result, f"expected ParseError, got {result.get('value')!r}"
    assert isinstance(result.get("exc"), ParseError), repr(result.get("exc"))


# ---- EB9: Generator end to end, one call --------------------------------------


@pytest.mark.parametrize("reply", [EB1, EB2, EB3], ids=["eb1", "eb2", "eb3_fenced"])
def test_eb9_generator_recovers_on_first_call(reply: str, _no_retry_sleep: list[float]) -> None:
    client = ScriptedLLMClient([reply])
    out = Generator(client).generate("Acme total 10.00")
    assert isinstance(out, InvoiceData)
    assert out.vendor_name == "Acme"
    assert out.total == 10.0
    assert len(client.calls) == 1
    assert _no_retry_sleep == []


# ---- EB10: Critic end to end, one call ----------------------------------------


def test_eb10_critic_recovers_on_first_call(_no_retry_sleep: list[float]) -> None:
    client = ScriptedLLMClient([EB10])
    out = Critic(client).critique(
        "Acme total 10.00", InvoiceData(vendor_name="Acme", total=10.0)
    )
    assert isinstance(out, CritiqueResult)
    assert out.overall_confidence == 0.9
    assert len(client.calls) == 1
    assert _no_retry_sleep == []


# ---- EB11: still-broken reply still retries -----------------------------------


def test_eb11_broken_reply_still_retries_and_raises(_no_retry_sleep: list[float]) -> None:
    client = ScriptedLLMClient([EB11_BAD, EB11_BAD])
    with pytest.raises(GeneratorError):
        Generator(client, max_retries=1).generate("x")
    assert len(client.calls) == 2


def test_eb11_counterpart_retry_recovers_with_good_second_reply() -> None:
    client = ScriptedLLMClient([EB11_BAD, EB1])
    out = Generator(client, max_retries=1).generate("Acme total 10.00")
    assert out.vendor_name == "Acme"
    assert len(client.calls) == 2
