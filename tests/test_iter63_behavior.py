"""Black-box behavioral tests for iter-63: validate builds no Generator client.

Covers Expected Behaviors 1-13 from the iter-63 spec:
  1-4  `build_default_pipeline(generator_enabled=...)` client counts / guards
  5-9  MCP `validate_tool` / `extract_tool` client counts and No-SDK envelopes
  10-11 CLI subprocess exit codes with No SDK
  12   factory kwargs recorded per entry point
  13   eval exit code + README documentation

Offline: the provider SDK is either a counting fake installed into
`sys.modules["anthropic"]`, or forced absent with `sys.modules["anthropic"] = None`.

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import reflect_mcp
import reflect_mcp.pipeline as pipeline_mod
from reflect_mcp import tools
from reflect_mcp.pipeline import ReflectivePipeline

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


def _critique() -> str:
    verdicts = [
        {"field": f, "confidence": 0.99, "grounded": True, "rationale": "matches source"}
        for f in DATA
    ]
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": 0.99, "summary": "ok"})


class _Counter:
    def __init__(self) -> None:
        self.built = 0
        self.calls: list[str | None] = []


def _fake_anthropic(counter: _Counter) -> types.ModuleType:
    """A fake `anthropic` module that counts client builds and model calls.

    The reply is chosen from the `model` kwarg: the Generator model gets DATA,
    the Critic model gets a high-confidence critique.
    """
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            model = kwargs.get("model")
            counter.calls.append(model)
            text = json.dumps(DATA) if model == GEN_MODEL else _critique()
            block = types.SimpleNamespace(text=text, type="text")
            return types.SimpleNamespace(content=[block], stop_reason="end_turn")

    class Anthropic:
        def __init__(self, *args, **kwargs) -> None:
            counter.built += 1
            self.messages = _Messages()

    mod.Anthropic = Anthropic
    return mod


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("REFLECT_PROVIDER", raising=False)
    monkeypatch.delenv("REFLECT_FIELD_THRESHOLD", raising=False)
    monkeypatch.delenv("REFLECT_OVERALL_THRESHOLD", raising=False)
    monkeypatch.setenv("REFLECT_GENERATOR_MODEL", GEN_MODEL)
    monkeypatch.setenv("REFLECT_CRITIC_MODEL", CRIT_MODEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    return monkeypatch


@pytest.fixture
def fake_sdk(env):
    counter = _Counter()
    env.setitem(sys.modules, "anthropic", _fake_anthropic(counter))
    return counter


@pytest.fixture
def no_sdk(env):
    env.setitem(sys.modules, "anthropic", None)
    return env


# ---------------------------------------------------------------------------
# EB-1 .. EB-4: the factory
# ---------------------------------------------------------------------------

def test_eb1_generator_disabled_builds_only_critic(fake_sdk):
    p = reflect_mcp.build_default_pipeline(generator_enabled=False)
    assert fake_sdk.built == 1
    assert isinstance(p, ReflectivePipeline)
    assert p.generator is None
    assert p.critic is not None
    assert fake_sdk.calls == []


@pytest.mark.parametrize("provider", [None, "bogus"])
def test_eb2_both_disabled_builds_zero_clients_no_sdk(no_sdk, provider):
    if provider is not None:
        no_sdk.setenv("REFLECT_PROVIDER", provider)
    p = reflect_mcp.build_default_pipeline(generator_enabled=False, critic_enabled=False)
    assert isinstance(p, ReflectivePipeline)
    assert p.generator is None
    assert p.critic is None


def test_eb2_both_disabled_builds_zero_clients_fake_sdk(fake_sdk):
    reflect_mcp.build_default_pipeline(generator_enabled=False, critic_enabled=False)
    assert fake_sdk.built == 0


def test_eb2_counterfactual_generator_enabled_still_needs_sdk(no_sdk):
    """Without generator_enabled=False, heuristics-only still needs a client."""
    with pytest.raises(Exception):
        reflect_mcp.build_default_pipeline(critic_enabled=False)


def test_eb3_default_build_unchanged(fake_sdk):
    from reflect_mcp.generator import Generator

    p = reflect_mcp.build_default_pipeline()
    assert fake_sdk.built == 2
    assert isinstance(p.generator, Generator)
    assert p.critic is not None


def test_eb3_signature():
    sig = inspect.signature(reflect_mcp.build_default_pipeline)
    params = sig.parameters
    ge = params["generator_enabled"]
    assert ge.kind is inspect.Parameter.KEYWORD_ONLY
    assert ge.default is True
    assert params["gate_config"].default is None
    assert params["max_retries"].default == 1
    assert params["critic_enabled"].default is True


def test_eb3_generator_enabled_is_keyword_only_at_runtime(fake_sdk):
    with pytest.raises(TypeError):
        reflect_mcp.build_default_pipeline(None, False)  # type: ignore[misc]


def test_eb4_extract_without_generator_raises_before_model_call(fake_sdk):
    from reflect_mcp.schema import InvoiceData

    p = reflect_mcp.build_default_pipeline(generator_enabled=False)
    with pytest.raises(RuntimeError) as exc:
        p.extract(TEXT)
    assert "without a Generator" in str(exc.value)
    assert fake_sdk.calls == []

    result = p.validate(TEXT, InvoiceData(**DATA))
    assert result.data.invoice_number == "INV-1"
    assert fake_sdk.calls == [CRIT_MODEL]


def test_eb4_extract_without_any_client_raises(no_sdk):
    p = reflect_mcp.build_default_pipeline(generator_enabled=False, critic_enabled=False)
    with pytest.raises(RuntimeError, match="without a Generator"):
        p.extract(TEXT)


# ---------------------------------------------------------------------------
# EB-5 .. EB-9: MCP tools
# ---------------------------------------------------------------------------

def test_eb5_validate_tool_builds_one_client_per_call(fake_sdk):
    statuses = [tools.validate_tool(TEXT, DATA)["status"] for _ in range(10)]
    assert fake_sdk.built == 10
    assert len(fake_sdk.calls) == 10
    assert set(fake_sdk.calls) == {CRIT_MODEL}
    assert len(set(statuses)) == 1
    assert statuses[0] != "error"


def test_eb5_validate_tool_status_matches_manual_pipeline(fake_sdk):
    """Status equals a pipeline built with both clients (HEAD's shape)."""
    from reflect_mcp.schema import InvoiceData

    got = tools.validate_tool(TEXT, DATA)
    ref = reflect_mcp.build_default_pipeline().validate(TEXT, InvoiceData(**DATA))
    ref_status = getattr(ref.status, "value", ref.status)
    assert got["status"] == ref_status
    assert got["overall_confidence"] == pytest.approx(ref.overall_confidence)


def test_eb6_validate_tool_heuristics_only_builds_zero(fake_sdk):
    outs = [tools.validate_tool(TEXT, DATA, heuristics_only=True) for _ in range(10)]
    assert fake_sdk.built == 0
    assert fake_sdk.calls == []
    assert all(o["status"] != "error" for o in outs)


@pytest.mark.parametrize("provider", [None, "bogus"])
def test_eb7_validate_heuristics_only_ok_without_sdk(no_sdk, provider):
    if provider is not None:
        no_sdk.setenv("REFLECT_PROVIDER", provider)
    out = tools.validate_tool(TEXT, DATA, heuristics_only=True)
    assert out["status"] == "ok", out
    assert "error_message" not in out


def test_eb8_validate_with_critic_no_sdk_still_errors(no_sdk):
    out = tools.validate_tool(TEXT, DATA)
    assert out["status"] == "error"
    assert "not installed" in out["error_message"]


def test_eb8_extract_heuristics_only_no_sdk_still_errors(no_sdk):
    out = tools.extract_tool(TEXT, heuristics_only=True)
    assert out["status"] == "error"
    assert out.get("error_message")


def test_eb8_validate_bogus_provider_errors(no_sdk):
    no_sdk.setenv("REFLECT_PROVIDER", "bogus")
    out = tools.validate_tool(TEXT, DATA)
    assert out["status"] == "error"
    assert "REFLECT_PROVIDER" in out["error_message"]


def test_eb9_extract_tool_still_builds_two_per_call(fake_sdk):
    for _ in range(10):
        tools.extract_tool(TEXT)
    assert fake_sdk.built == 20


# ---------------------------------------------------------------------------
# EB-10 .. EB-11: CLI subprocess, No SDK
# ---------------------------------------------------------------------------

_BLOCK_SDK = (
    "import sys, runpy; sys.modules['anthropic'] = None; "
    "sys.argv = ['reflect_mcp'] + sys.argv[1:]; "
    "runpy.run_module('reflect_mcp', run_name='__main__')"
)


def _cli(args: list[str], provider: str | None = None):
    src = str(REPO_ROOT / "src")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": src}
    if provider is not None:
        env["REFLECT_PROVIDER"] = provider
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_SDK, *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=60,
    )


def _text_file(tmp_path) -> str:
    path = tmp_path / "invoice.txt"
    path.write_text(TEXT)
    return str(path)


def _last_json(stdout: str) -> dict:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    assert lines, "no stdout"
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return json.loads(lines[-1])


@pytest.mark.parametrize("provider", [None, "bogus"])
def test_eb10_cli_validate_heuristics_only_no_sdk_exits_0(tmp_path, provider):
    proc = _cli(
        ["validate", "--heuristics-only", "--data", json.dumps(DATA), "--file", _text_file(tmp_path)],
        provider=provider,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _last_json(proc.stdout)["status"] == "ok"


def test_eb11_cli_validate_with_critic_no_sdk_exits_3(tmp_path):
    proc = _cli(["validate", "--data", json.dumps(DATA), "--file", _text_file(tmp_path)])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert _last_json(proc.stdout)["status"] == "error"


def test_eb11_cli_extract_heuristics_only_no_sdk_exits_3(tmp_path):
    proc = _cli(["extract", "--heuristics-only", "--file", _text_file(tmp_path)])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert _last_json(proc.stdout)["status"] == "error"


# ---------------------------------------------------------------------------
# EB-12: factory kwargs per entry point
# ---------------------------------------------------------------------------

@pytest.fixture
def recorder(fake_sdk, monkeypatch):
    """Record kwargs passed to build_default_pipeline, then delegate to the real one."""
    real = pipeline_mod.build_default_pipeline
    seen: list[dict] = []

    def _rec(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "build_default_pipeline", _rec)
    # If an entry point imported the name directly, patch that binding too.
    import importlib

    for name in ("reflect_mcp.tools", "reflect_mcp.cli", "reflect_mcp.__main__"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if getattr(mod, "build_default_pipeline", None) is real:
            monkeypatch.setattr(mod, "build_default_pipeline", _rec)
    return seen


def _run_main(monkeypatch, argv: list[str], stdin: str | None = None) -> int:
    import io

    from reflect_mcp.__main__ import main

    monkeypatch.setattr(sys, "argv", ["reflect-mcp", *argv])
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    try:
        rc = main()
    except SystemExit as e:  # argparse / explicit exit
        rc = e.code
    return 0 if rc is None else rc


@pytest.mark.parametrize("ho", [False, True])
def test_eb12_cli_validate_passes_generator_disabled(recorder, monkeypatch, tmp_path, capsys, ho):
    argv = ["validate", "--data", json.dumps(DATA), "--file", _text_file(tmp_path)]
    if ho:
        argv.append("--heuristics-only")
    _run_main(monkeypatch, argv)
    assert recorder, "factory never called"
    for kw in recorder:
        assert kw.get("generator_enabled", "MISSING") is False
        assert kw.get("critic_enabled", True) is (not ho)


@pytest.mark.parametrize("ho", [False, True])
def test_eb12_mcp_validate_passes_generator_disabled(recorder, ho):
    tools.validate_tool(TEXT, DATA, heuristics_only=ho)
    assert len(recorder) == 1
    assert recorder[0]["generator_enabled"] is False
    assert recorder[0].get("critic_enabled", True) is (not ho)


@pytest.mark.parametrize("ho", [False, True])
def test_eb12_mcp_extract_does_not_pass_generator_enabled(recorder, ho):
    tools.extract_tool(TEXT, heuristics_only=ho)
    assert len(recorder) == 1
    assert "generator_enabled" not in recorder[0]
    assert recorder[0].get("critic_enabled", True) is (not ho)


@pytest.mark.parametrize("ho", [False, True])
def test_eb12_cli_extract_does_not_pass_generator_enabled(recorder, monkeypatch, tmp_path, capsys, ho):
    argv = ["extract", "--file", _text_file(tmp_path)]
    if ho:
        argv.append("--heuristics-only")
    _run_main(monkeypatch, argv)
    assert recorder, "factory never called"
    for kw in recorder:
        assert "generator_enabled" not in kw
        assert kw.get("critic_enabled", True) is (not ho)


@pytest.mark.parametrize("ho", [False, True])
def test_eb12_cli_extract_batch_does_not_pass_generator_enabled(recorder, monkeypatch, tmp_path, capsys, ho):
    path = tmp_path / "batch.jsonl"
    path.write_text(json.dumps({"text": TEXT}) + "\n" + json.dumps({"text": TEXT}) + "\n")
    argv = ["extract", "--batch", "--file", str(path)]
    if ho:
        argv.append("--heuristics-only")
    _run_main(monkeypatch, argv)
    assert recorder, "factory never called"
    for kw in recorder:
        assert "generator_enabled" not in kw
        assert kw.get("critic_enabled", True) is (not ho)


# ---------------------------------------------------------------------------
# EB-13: eval exit code + README
# ---------------------------------------------------------------------------

def test_eb13_eval_still_passes():
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "eval" / "run_eval.py")],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip()


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text()


def test_eb13_readme_documents_generator_enabled():
    readme = _readme()
    assert "generator_enabled=False" in readme
    assert "validate-only: builds no Generator client" in readme


def test_eb13_readme_heuristics_only_needs_no_sdk_for_validate():
    readme = _readme()
    para = next(p for p in readme.split("\n\n") if p.startswith("**Heuristics-only mode**"))
    assert "validate" in para
    assert "no provider SDK" in para
    assert "API key" in para
