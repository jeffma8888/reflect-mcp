"""Black-box behavioral tests for iter-118: MCP tools memoise one provider client per role.

Covers Expected Behaviors 1-10 from the iter-118 spec:
  1   20 extracts -> 2 builds (1 generator, 1 critic); model calls unchanged; same dicts
  2   20 validates -> 1 critic build; after extract, validate builds 0
  3   heuristics_only builds 0 critic clients
  4   8 barrier-started threads -> exactly 1 critic build, same non-error status
  5   env selection (critic/generator model, provider, region) re-keys the memo
  6   failed builds are not cached
  7   a monkeypatched `providers.build_client` is used immediately (factory identity keyed)
  8   `build_default_pipeline` default unchanged; `client_factory=f` called per enabled role
  9   conftest autouse clears `tools._CLIENT_MEMO`
  10  README MCP-section sentence; no secret in the memo key

Offline: either a counting `providers.build_client` fake returning stub LLM clients,
or a counting fake `anthropic` module in `sys.modules` (the iter63 pattern).
No wall-clock assertions: every bound is a build/call count.

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import reflect_mcp
from reflect_mcp import providers, tools

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT = "Invoice INV-1\nVendor: Acme\nDate: 2024-01-05\nTotal: 10.00 USD\n"
DATA = {
    "invoice_number": "INV-1",
    "vendor_name": "Acme",
    "invoice_date": "2024-01-05",
    "total": 10.0,
    "currency": "USD",
}
GEN_MODEL = "gen-model-x"
CRIT_MODEL = "crit-model-y"
SECRET = "sk-iter118-sentinel-not-a-real-key"


def _critique() -> str:
    verdicts = [
        {"field": f, "confidence": 0.99, "grounded": True, "rationale": "matches source"}
        for f in DATA
    ]
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": 0.99, "summary": "ok"})


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _StubClient:
    """An LLM client that answers by role and records every call."""

    def __init__(self, role: str, log: list[str], tag: str = "") -> None:
        self.role = role
        self.log = log
        self.tag = tag

    def complete(self, *, system: str, user: str) -> str:
        self.log.append(f"{self.tag}{self.role}")
        return json.dumps(DATA) if self.role == "generator" else _critique()


class _Factory:
    """A counting stand-in for `providers.build_client(role)`."""

    def __init__(self, tag: str = "", sleep: float = 0.0, fail_first: int = 0) -> None:
        self.tag = tag
        self.sleep = sleep
        self.fail_first = fail_first
        self.roles: list[str] = []
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, role, *args, **kwargs):
        if self.sleep:
            time.sleep(self.sleep)  # widen the race window; never asserted on
        with self._lock:
            if self.fail_first > 0:
                self.fail_first -= 1
                raise RuntimeError("anthropic SDK not installed (fake build failure)")
            self.roles.append(role)
        return _StubClient(role, self.calls, self.tag)


class _Counter:
    def __init__(self) -> None:
        self.built = 0
        self.models: list[str | None] = []


def _fake_anthropic(counter: _Counter) -> types.ModuleType:
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            model = kwargs.get("model")
            counter.models.append(model)
            text = json.dumps(DATA) if model == GEN_MODEL else _critique()
            block = types.SimpleNamespace(text=text, type="text")
            return types.SimpleNamespace(content=[block], stop_reason="end_turn")

    class Anthropic:
        def __init__(self, *args, **kwargs) -> None:
            counter.built += 1
            self.messages = _Messages()

    mod.Anthropic = Anthropic
    return mod


def _memo():
    memo = getattr(tools, "_CLIENT_MEMO", None)
    assert memo is not None, "tools._CLIENT_MEMO does not exist"
    return memo


@pytest.fixture
def env(monkeypatch):
    for name in ("REFLECT_PROVIDER", "AWS_REGION", "REFLECT_FIELD_THRESHOLD", "REFLECT_OVERALL_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REFLECT_GENERATOR_MODEL", GEN_MODEL)
    monkeypatch.setenv("REFLECT_CRITIC_MODEL", CRIT_MODEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    memo = getattr(tools, "_CLIENT_MEMO", None)
    if memo is not None:
        memo.clear()
    return monkeypatch


@pytest.fixture
def factory(env):
    f = _Factory()
    env.setattr(providers, "build_client", f)
    return f


@pytest.fixture
def fake_sdk(env):
    counter = _Counter()
    env.setitem(sys.modules, "anthropic", _fake_anthropic(counter))
    return counter


def _clear() -> None:
    memo = getattr(tools, "_CLIENT_MEMO", None)
    if memo is not None:
        memo.clear()


# ---------------------------------------------------------------------------
# EB-1: extract
# ---------------------------------------------------------------------------

def test_eb1_twenty_extracts_build_two_clients(factory):
    outs = [tools.extract_tool(TEXT) for _ in range(20)]
    assert sorted(factory.roles) == ["critic", "generator"]
    # Model calls are not memoised: 2 per extract (generator + critic).
    assert len(factory.calls) == 40
    assert factory.calls.count("generator") == 20
    assert factory.calls.count("critic") == 20
    assert all(o["status"] != "error" for o in outs), outs[0]


def test_eb1_fake_sdk_twenty_extracts_build_two(fake_sdk):
    for _ in range(20):
        tools.extract_tool(TEXT)
    assert fake_sdk.built == 2
    assert fake_sdk.models.count(GEN_MODEL) == 20
    assert fake_sdk.models.count(CRIT_MODEL) == 20


def test_eb1_memoised_dicts_match_fresh_build_dicts(factory):
    """Results with the memo equal results when every call builds fresh (HEAD shape)."""
    memo_outs = [tools.extract_tool(TEXT) for _ in range(3)]
    fresh_outs = []
    for _ in range(3):
        _clear()
        fresh_outs.append(tools.extract_tool(TEXT))
    # trace_id is per-call by design; everything else must be identical.
    strip = lambda d: {k: v for k, v in d.items() if k != "trace_id"}  # noqa: E731
    assert [strip(d) for d in memo_outs] == [strip(d) for d in fresh_outs]
    assert len(strip(memo_outs[0])) >= 5
    assert memo_outs[0]["status"] != "error"


# ---------------------------------------------------------------------------
# EB-2: validate
# ---------------------------------------------------------------------------

def test_eb2_twenty_validates_build_one_critic(factory):
    outs = [tools.validate_tool(TEXT, DATA) for _ in range(20)]
    assert factory.roles == ["critic"]
    assert factory.calls == ["critic"] * 20
    assert len({o["status"] for o in outs}) == 1
    assert outs[0]["status"] != "error"


def test_eb2_validate_after_extract_builds_zero(factory):
    tools.extract_tool(TEXT)
    assert sorted(factory.roles) == ["critic", "generator"]
    for _ in range(5):
        out = tools.validate_tool(TEXT, DATA)
        assert out["status"] != "error"
    assert sorted(factory.roles) == ["critic", "generator"]  # no new build


# ---------------------------------------------------------------------------
# EB-3: heuristics_only
# ---------------------------------------------------------------------------

def test_eb3_validate_heuristics_only_builds_zero(factory):
    outs = [tools.validate_tool(TEXT, DATA, heuristics_only=True) for _ in range(5)]
    assert factory.roles == []
    assert factory.calls == []
    assert all(o["status"] != "error" for o in outs)


def test_eb3_extract_heuristics_only_builds_no_critic(factory):
    outs = [tools.extract_tool(TEXT, heuristics_only=True) for _ in range(5)]
    assert "critic" not in factory.roles
    assert factory.roles == ["generator"]
    assert "critic" not in factory.calls
    assert all(o["status"] != "error" for o in outs)


# ---------------------------------------------------------------------------
# EB-4: concurrency
# ---------------------------------------------------------------------------

def test_eb4_eight_barrier_threads_build_one_critic(env):
    f = _Factory(sleep=0.02)
    env.setattr(providers, "build_client", f)
    n = 8
    barrier = threading.Barrier(n)
    results: list[dict] = [None] * n  # type: ignore[list-item]
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=30)
            results[i] = tools.validate_tool(TEXT, DATA)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert f.roles == ["critic"], f.roles
    assert len(f.calls) == n
    statuses = {r["status"] for r in results}
    assert len(statuses) == 1, statuses
    assert statuses != {"error"}


# ---------------------------------------------------------------------------
# EB-5: env selection invalidates
# ---------------------------------------------------------------------------

def test_eb5_critic_model_switch_rebuilds_and_switch_back_reuses(fake_sdk, env):
    env.setenv("REFLECT_CRITIC_MODEL", "m1")
    tools.validate_tool(TEXT, DATA)
    assert fake_sdk.built == 1
    assert fake_sdk.models == ["m1"]

    env.setenv("REFLECT_CRITIC_MODEL", "m2")
    tools.validate_tool(TEXT, DATA)
    assert fake_sdk.built == 2
    assert fake_sdk.models[-1] == "m2"

    env.setenv("REFLECT_CRITIC_MODEL", "m1")
    tools.validate_tool(TEXT, DATA)
    assert fake_sdk.built == 2  # m1 client reused
    assert fake_sdk.models[-1] == "m1"


def test_eb5_provider_change_rebuilds(factory, env):
    tools.validate_tool(TEXT, DATA)
    assert factory.roles == ["critic"]
    env.setenv("REFLECT_PROVIDER", "openai")
    tools.validate_tool(TEXT, DATA)
    assert factory.roles == ["critic", "critic"]


def test_eb5_provider_case_and_default_share_a_key(factory, env):
    """Key uses REFLECT_PROVIDER lowercased, default anthropic."""
    tools.validate_tool(TEXT, DATA)
    env.setenv("REFLECT_PROVIDER", "ANTHROPIC")
    tools.validate_tool(TEXT, DATA)
    env.setenv("REFLECT_PROVIDER", "anthropic")
    tools.validate_tool(TEXT, DATA)
    assert factory.roles == ["critic"]


def test_eb5_generator_model_change_rebuilds_generator_only(factory, env):
    tools.extract_tool(TEXT)
    assert sorted(factory.roles) == ["critic", "generator"]
    env.setenv("REFLECT_GENERATOR_MODEL", "gen-model-z")
    tools.extract_tool(TEXT)
    assert factory.roles.count("generator") == 2
    assert factory.roles.count("critic") == 1


def test_eb5_aws_region_change_rebuilds(factory, env):
    env.setenv("AWS_REGION", "us-east-1")
    tools.validate_tool(TEXT, DATA)
    env.setenv("AWS_REGION", "eu-west-1")
    tools.validate_tool(TEXT, DATA)
    assert factory.roles == ["critic", "critic"]


# ---------------------------------------------------------------------------
# EB-6: failures not cached
# ---------------------------------------------------------------------------

def test_eb6_no_sdk_error_then_recovers(env):
    env.setitem(sys.modules, "anthropic", None)
    out = tools.validate_tool(TEXT, DATA)
    assert out["status"] == "error"
    assert "not installed" in out["error_message"]
    assert len(_memo()) == 0

    counter = _Counter()
    env.setitem(sys.modules, "anthropic", _fake_anthropic(counter))
    out2 = tools.validate_tool(TEXT, DATA)
    assert out2["status"] != "error", out2
    assert counter.built == 1


def test_eb6_raising_factory_not_cached(env):
    f = _Factory(fail_first=1)
    env.setattr(providers, "build_client", f)
    out = tools.validate_tool(TEXT, DATA)
    assert out["status"] == "error"
    assert "not installed" in out["error_message"]
    assert f.roles == []
    out2 = tools.validate_tool(TEXT, DATA)
    assert out2["status"] != "error", out2
    assert f.roles == ["critic"]
    out3 = tools.validate_tool(TEXT, DATA)
    assert out3["status"] != "error"
    assert f.roles == ["critic"]


# ---------------------------------------------------------------------------
# EB-7: factory identity keyed
# ---------------------------------------------------------------------------

def test_eb7_replaced_factory_is_used_next_call(env):
    a = _Factory(tag="A:")
    env.setattr(providers, "build_client", a)
    tools.validate_tool(TEXT, DATA)
    assert a.roles == ["critic"]
    assert a.calls == ["A:critic"]

    b = _Factory(tag="B:")
    env.setattr(providers, "build_client", b)
    tools.validate_tool(TEXT, DATA)
    assert b.roles == ["critic"]
    assert b.calls == ["B:critic"]
    assert a.calls == ["A:critic"]  # A's memoised client was not used again


# ---------------------------------------------------------------------------
# EB-8: build_default_pipeline unchanged by default; client_factory seam
# ---------------------------------------------------------------------------

def test_eb8_signature_client_factory_defaults_none():
    params = inspect.signature(reflect_mcp.build_default_pipeline).parameters
    assert "client_factory" in params
    assert params["client_factory"].default is None


def test_eb8_default_builds_fresh_each_call(fake_sdk):
    for _ in range(3):
        reflect_mcp.build_default_pipeline()
    assert fake_sdk.built == 6


def test_eb8_default_goes_through_providers_build_client(factory):
    reflect_mcp.build_default_pipeline()
    reflect_mcp.build_default_pipeline()
    assert sorted(factory.roles) == ["critic", "critic", "generator", "generator"]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, ["critic", "generator"]),
        ({"generator_enabled": False}, ["critic"]),
        ({"critic_enabled": False}, ["generator"]),
        ({"generator_enabled": False, "critic_enabled": False}, []),
    ],
)
def test_eb8_client_factory_called_for_enabled_roles_only(factory, kwargs, expected):
    seen: list[str] = []

    def f(role):
        seen.append(role)
        return _StubClient(role, [])

    reflect_mcp.build_default_pipeline(client_factory=f, **kwargs)
    assert sorted(seen) == expected
    assert factory.roles == []  # providers.build_client bypassed


# ---------------------------------------------------------------------------
# EB-9: conftest autouse memo clear
# ---------------------------------------------------------------------------

def test_eb9_memo_exists_and_is_clearable():
    memo = _memo()
    assert hasattr(memo, "clear")


def test_eb9a_populate_memo(factory):
    tools.validate_tool(TEXT, DATA)
    assert len(_memo()) >= 1


def test_eb9b_memo_empty_at_test_start():
    # Runs after eb9a in file order; the suite-wide autouse fixture must have cleared it.
    assert len(_memo()) == 0


def test_eb9_conftest_declares_autouse_memo_clear():
    text = (REPO_ROOT / "tests" / "conftest.py").read_text()
    assert "_CLIENT_MEMO" in text
    assert re.search(r"autouse=True\)\s*\ndef \w+\([^)]*\)[^:]*:\s*\n(\s+\"\"\"[^\n]*\n)?\s+tools\._CLIENT_MEMO\.clear\(\)", text)


# ---------------------------------------------------------------------------
# EB-10: README sentence; no secret in the key
# ---------------------------------------------------------------------------

def _readme_sections() -> dict[str, str]:
    text = (REPO_ROOT / "README.md").read_text()
    sections: dict[str, str] = {}
    current = "_preamble"
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = ""
        else:
            sections[current] = sections.get(current, "") + line + "\n"
    return sections


_REUSE_RE = re.compile(r"reuses?\b[^.]*\bacross tool calls", re.IGNORECASE)


def test_eb10_readme_mcp_section_documents_reuse_and_restart():
    sections = _readme_sections()
    hits = [name for name, body in sections.items() if _REUSE_RE.search(body)]
    assert hits, "README never says the server reuses a provider client across tool calls"
    assert all("claude desktop" in name.lower() for name in hits), hits  # §4, the MCP section
    assert not any(name.startswith("9.") for name in hits)
    body = sections[hits[0]]
    sentence = next(s for s in re.split(r"(?<=[.!?])\s+", body) if _REUSE_RE.search(s))
    low = sentence.lower()
    assert "provider client" in low
    assert "restart" in low
    assert "credential" in low or "api key" in low


def test_eb10_api_key_rotation_does_not_rebuild(factory, env):
    tools.validate_tool(TEXT, DATA)
    env.setenv("ANTHROPIC_API_KEY", SECRET + "-rotated")
    env.setenv("OPENAI_API_KEY", SECRET)
    tools.validate_tool(TEXT, DATA)
    assert factory.roles == ["critic"]


def test_eb10_memo_key_holds_no_secret(factory, env):
    env.setenv("OPENAI_API_KEY", SECRET)
    env.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    tools.extract_tool(TEXT)
    memo = _memo()
    keys = list(memo.keys()) if hasattr(memo, "keys") else list(memo)
    assert keys
    assert SECRET not in repr(keys)
