"""Black-box behavioral tests for iter-92: one lazy-SDK guard in the provider seam.

Covers the Expected Behaviors from the iter-92 spec:
  1-3. Missing SDK -> RuntimeError with the exact install hint, __cause__ ImportError
  4.   Each provider's SDK is constructed once, with the documented arguments
  5.   build_client dispatch per REFLECT_PROVIDER (case-folded, default anthropic)
  6.   Unknown provider -> exact ValueError, no SDK import attempted
  7.   _CLIENTS dispatch table key set equals _DEFAULTS
  8.   One guard, structurally (ast walk of the provider module)
  9.   Unchanged surface -- covered by the untouched iter-03 / iter-63 modules
       and the make eval tests already in the suite (not duplicated here)
  10.  This module runs no subprocess and reads no VCS revision (ast self-check)

Isolation contract: tests drive the public provider interface and inspect
observable output/attributes. EB8 parses the module file only because the
spec's behavior is itself structural; the implementation was not read by hand.
All tests run offline with no provider key set.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reflect_mcp import providers as P

SDKS = ("anthropic", "openai", "boto3")
ENV_VARS = (
    "REFLECT_PROVIDER",
    "REFLECT_GENERATOR_MODEL",
    "REFLECT_CRITIC_MODEL",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)

HINTS = {
    "anthropic": (
        "Anthropic provider selected but the `anthropic` package is not installed. "
        "Run: pip install 'reflect-mcp[anthropic]'"
    ),
    "openai": (
        "OpenAI provider selected but the `openai` package is not installed. "
        "Run: pip install 'reflect-mcp[openai]'"
    ),
    "bedrock": (
        "Bedrock provider selected but the `boto3` package is not installed. "
        "Run: pip install 'reflect-mcp[bedrock]'"
    ),
}
UNKNOWN_MSG = "Unknown REFLECT_PROVIDER='bogus'. Choose one of: anthropic, openai, bedrock."

CLASS_FOR = {
    "anthropic": "AnthropicClient",
    "openai": "OpenAIClient",
    "bedrock": "BedrockClient",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _fake_sdks():
    """A fresh MagicMock module per SDK name."""
    return {name: MagicMock(name=f"fake_{name}") for name in SDKS}


def _no_sdks():
    return {name: None for name in SDKS}


# ---------------------------------------------------------------------------
# EB1-3: SDK missing -> exact install hint, chained from ImportError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "provider, sdk",
    [("anthropic", "anthropic"), ("openai", "openai"), ("bedrock", "boto3")],
)
def test_eb1_3_missing_sdk_raises_exact_install_hint(provider, sdk):
    cls = getattr(P, CLASS_FOR[provider])
    with patch.dict("sys.modules", {sdk: None}):
        with pytest.raises(RuntimeError) as excinfo:
            cls(model="m")
    assert str(excinfo.value) == HINTS[provider]
    assert isinstance(excinfo.value.__cause__, ImportError), (
        f"{provider}: __cause__ is {type(excinfo.value.__cause__).__name__}, "
        "expected ImportError (raise ... from exc)"
    )


def test_eb3_bedrock_wording_unified_with_other_two():
    """The three hints share one template: '<Name> provider selected but the
    `<pkg>` package is not installed. Run: pip install 'reflect-mcp[<extra>]'"""
    msgs = {}
    for provider, sdk in (("anthropic", "anthropic"), ("openai", "openai"), ("bedrock", "boto3")):
        cls = getattr(P, CLASS_FOR[provider])
        with patch.dict("sys.modules", {sdk: None}):
            with pytest.raises(RuntimeError) as excinfo:
                cls(model="m")
        msgs[provider] = str(excinfo.value)
    templ = {
        p: m.replace(f"`{s}`", "`X`").replace(f"[{p}]", "[E]").split(" provider", 1)[1]
        for (p, s), m in zip(
            (("anthropic", "anthropic"), ("openai", "openai"), ("bedrock", "boto3")),
            msgs.values(),
        )
    }
    assert len(set(templ.values())) == 1, templ
    assert "but `boto3` is not installed" not in msgs["bedrock"]


# ---------------------------------------------------------------------------
# EB4: each SDK constructed once, with HEAD-equal arguments
# ---------------------------------------------------------------------------

def test_eb4_anthropic_constructed_once_no_args():
    fakes = _fake_sdks()
    with patch.dict("sys.modules", fakes):
        client = P.AnthropicClient(model="m")
    ctor = fakes["anthropic"].Anthropic
    assert ctor.call_count == 1
    assert ctor.call_args.args == () and ctor.call_args.kwargs == {}
    assert client._client is ctor.return_value
    assert client.model == "m"


def test_eb4_openai_constructed_once_no_args():
    fakes = _fake_sdks()
    with patch.dict("sys.modules", fakes):
        client = P.OpenAIClient(model="m")
    ctor = fakes["openai"].OpenAI
    assert ctor.call_count == 1
    assert ctor.call_args.args == () and ctor.call_args.kwargs == {}
    assert client.model == "m"


def test_eb4_bedrock_region_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-3")
    fakes = _fake_sdks()
    with patch.dict("sys.modules", fakes):
        P.BedrockClient(model="m")
    ctor = fakes["boto3"].client
    assert ctor.call_count == 1
    assert ctor.call_args.args == ("bedrock-runtime",)
    assert ctor.call_args.kwargs == {"region_name": "eu-west-3"}


def test_eb4_bedrock_region_default_when_env_unset():
    fakes = _fake_sdks()
    with patch.dict("sys.modules", fakes):
        P.BedrockClient(model="m")
    ctor = fakes["boto3"].client
    assert ctor.call_count == 1
    assert ctor.call_args.args == ("bedrock-runtime",)
    assert ctor.call_args.kwargs == {"region_name": "us-east-1"}


def test_eb4_other_sdks_untouched_per_provider():
    """Building one provider constructs nothing from the other SDKs."""
    fakes = _fake_sdks()
    with patch.dict("sys.modules", fakes):
        P.AnthropicClient(model="m")
    assert fakes["openai"].OpenAI.call_count == 0
    assert fakes["boto3"].client.call_count == 0


# ---------------------------------------------------------------------------
# EB5: build_client dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "env_value, provider",
    [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("bedrock", "bedrock"),
        ("ANTHROPIC", "anthropic"),
    ],
)
def test_eb5_build_client_dispatch(monkeypatch, env_value, provider):
    monkeypatch.setenv("REFLECT_PROVIDER", env_value)
    with patch.dict("sys.modules", _fake_sdks()):
        client = P.build_client("generator")
    expected_cls = getattr(P, CLASS_FOR[provider])
    assert type(client) is expected_cls
    assert client.model == P._DEFAULTS[provider]["generator"]


@pytest.mark.parametrize("provider", ["anthropic", "openai", "bedrock"])
def test_eb5_critic_model_env_override(monkeypatch, provider):
    monkeypatch.setenv("REFLECT_PROVIDER", provider)
    monkeypatch.setenv("REFLECT_CRITIC_MODEL", "x")
    with patch.dict("sys.modules", _fake_sdks()):
        client = P.build_client("critic")
    assert type(client) is getattr(P, CLASS_FOR[provider])
    assert client.model == "x"


def test_eb5_unset_provider_defaults_to_anthropic():
    with patch.dict("sys.modules", _fake_sdks()):
        client = P.build_client("generator")
    assert type(client) is P.AnthropicClient
    assert client.model == P._DEFAULTS["anthropic"]["generator"]


# ---------------------------------------------------------------------------
# EB6: unknown provider
# ---------------------------------------------------------------------------

def test_eb6_unknown_provider_exact_value_error(monkeypatch):
    monkeypatch.setenv("REFLECT_PROVIDER", "bogus")
    with pytest.raises(ValueError) as excinfo:
        P.build_client("generator")
    assert str(excinfo.value) == UNKNOWN_MSG


def test_eb6_unknown_provider_imports_no_sdk(monkeypatch):
    monkeypatch.setenv("REFLECT_PROVIDER", "bogus")
    boom = AssertionError("an SDK import was attempted for an unknown provider")
    with patch.dict("sys.modules", _no_sdks()), patch(
        "importlib.import_module", side_effect=boom
    ) as imp:
        with pytest.raises(ValueError) as excinfo:
            P.build_client("generator")
    assert str(excinfo.value) == UNKNOWN_MSG
    assert imp.call_count == 0


# ---------------------------------------------------------------------------
# EB7: dispatch table equals defaults table
# ---------------------------------------------------------------------------

def test_eb7_clients_keys_equal_defaults_keys():
    assert isinstance(P._CLIENTS, dict)
    assert set(P._CLIENTS) == set(P._DEFAULTS)
    assert set(P._CLIENTS) == {"anthropic", "openai", "bedrock"}


def test_eb7_clients_values_are_llm_clients():
    for key, cls in P._CLIENTS.items():
        assert inspect.isclass(cls), key
        assert callable(getattr(cls, "complete", None)), key
        assert cls is getattr(P, CLASS_FOR[key]), key
        with patch.dict("sys.modules", _fake_sdks()):
            instance = cls(model="m")
        assert isinstance(instance, P.LLMClient), key


# ---------------------------------------------------------------------------
# EB8: one guard, structurally
# ---------------------------------------------------------------------------

PROVIDERS_PATH = Path(P.__file__)
PROVIDERS_SRC = PROVIDERS_PATH.read_text(encoding="utf-8")
PROVIDERS_TREE = ast.parse(PROVIDERS_SRC)


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    t = handler.type
    elts = t.elts if isinstance(t, ast.Tuple) else [t] if t is not None else []
    return {e.id for e in elts if isinstance(e, ast.Name)}


def test_eb8_exactly_one_import_error_handler():
    handlers = [
        n for n in ast.walk(PROVIDERS_TREE)
        if isinstance(n, ast.ExceptHandler) and "ImportError" in _handler_names(n)
    ]
    assert len(handlers) == 1, f"found {len(handlers)} ImportError handlers"


def test_eb8_one_require_function_calling_import_module():
    funcs = [
        n for n in ast.walk(PROVIDERS_TREE)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_require"
    ]
    assert len(funcs) == 1, f"found {len(funcs)} functions named _require"
    params = [a.arg for a in funcs[0].args.args]
    assert len(params) == 3, params

    def _is_import_module(call: ast.Call) -> bool:
        f = call.func
        return (
            isinstance(f, ast.Attribute)
            and f.attr == "import_module"
            and isinstance(f.value, ast.Name)
            and f.value.id == "importlib"
        )

    calls = [n for n in ast.walk(funcs[0]) if isinstance(n, ast.Call) and _is_import_module(n)]
    assert calls, "_require does not call importlib.import_module"


def test_eb8_no_direct_sdk_imports_or_provider_if_chain():
    sdk_imports = []
    for n in ast.walk(PROVIDERS_TREE):
        if isinstance(n, ast.Import):
            sdk_imports += [a.name for a in n.names if a.name.split(".")[0] in SDKS]
        elif isinstance(n, ast.ImportFrom) and n.module:
            if n.module.split(".")[0] in SDKS:
                sdk_imports.append(n.module)
    assert sdk_imports == [], sdk_imports
    for needle in ("import anthropic", "from openai import", "import boto3", "if provider =="):
        assert needle not in PROVIDERS_SRC, f"provider module still contains {needle!r}"


def test_eb8_line_count_within_cap():
    n_lines = len(PROVIDERS_SRC.splitlines())
    assert n_lines <= 131, f"providers.py has {n_lines} lines (cap 131)"


def test_eb8_require_is_the_only_guard_each_client_calls_it():
    """Each client __init__ calls _require exactly once."""
    for cls_name in CLASS_FOR.values():
        cls_node = next(
            n for n in PROVIDERS_TREE.body
            if isinstance(n, ast.ClassDef) and n.name == cls_name
        )
        init = next(
            n for n in cls_node.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        calls = [
            n for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_require"
        ]
        assert len(calls) == 1, f"{cls_name}.__init__ calls _require {len(calls)} times"


# ---------------------------------------------------------------------------
# EB9: unchanged surface -- reuse of the import guard keeps sys.modules stubs
# ---------------------------------------------------------------------------

def test_eb9_validate_style_no_sdk_still_mentions_not_installed(monkeypatch):
    """With no SDK at all, every provider path still says 'not installed'."""
    for provider in ("anthropic", "openai", "bedrock"):
        monkeypatch.setenv("REFLECT_PROVIDER", provider)
        with patch.dict("sys.modules", _no_sdks()):
            with pytest.raises(RuntimeError) as excinfo:
                P.build_client("generator")
        assert str(excinfo.value) == HINTS[provider]


# ---------------------------------------------------------------------------
# EB10: this module is safe in a committed clone (no subprocess, no VCS revs)
# ---------------------------------------------------------------------------

def test_eb10_self_check_no_subprocess_no_vcs_revs():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    rev_token = "HE" + "AD"
    vcs_verbs = tuple("g" + "it " + v for v in ("archive", "show", "diff", "log", "rev-parse"))
    banned_mods = {"sub" + "process", "o" + "s"}
    problems = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            problems += [f"imports {a.name}" for a in n.names if a.name in banned_mods]
        elif isinstance(n, ast.ImportFrom) and n.module in banned_mods:
            problems.append(f"imports from {n.module}")
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            if n.value == rev_token or n.value.startswith(vcs_verbs):
                problems.append(f"literal {n.value!r} at line {n.lineno}")
            if n.value == "g" + "it":
                problems.append(f"vcs executable literal at line {n.lineno}")
    assert problems == [], "module is not committed-clone safe: " + "; ".join(problems)


def test_eb10_importlib_is_real_module():
    """Sanity: patching importlib.import_module (EB6) targets the stdlib."""
    assert importlib.import_module("json") is sys.modules["json"]
