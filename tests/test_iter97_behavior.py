"""Black-box behavioral tests for iter-97: ``reflect-mcp --mcp-config`` prints a
paste-ready MCP client-config block for this install (roadmap #97).

Covers Expected Behaviors 1-10 from the iter-97 spec:
  1  stdout is JSON with exact shape {"mcpServers": {"reflect-mcp": {command,args,env}}},
     exit 0, empty stderr
  2  command == sys.executable, args == ["-m", "reflect_mcp.server"], and that
     interpreter can import reflect_mcp.server
  3  default env (hermetic) is exactly REFLECT_PROVIDER=anthropic + key placeholder, in order
  4  provider-specific key placeholder (openai), lowercased bedrock with no *_API_KEY
  5  exported fake secrets never appear in stdout/stderr at any provider
  6  set REFLECT_* echo vars appear verbatim in spec order; unrelated/empty vars omitted
  7  unknown provider -> exit 2, empty stdout, stderr names bogus + the valid providers
  8  --help lists --mcp-config; --version unchanged; --mcp-config with a verb runs no verb
  9  footprint: cli.py stays 306 lines; frozen eval stdout byte-identical to HEAD
 10  README section 4 mentions --mcp-config and stays publication-safe

HEAD constants (frozen-eval stdout digest, --version text) were measured from the
pre-change commit's product; no git command runs at test time. Fake secrets are
built in-test and never hit any network.

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests._golden import frozen_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = "<your-api-key>"
ECHO_ORDER = [
    "REFLECT_PROVIDER",
    "REFLECT_GENERATOR_MODEL",
    "REFLECT_CRITIC_MODEL",
    "REFLECT_FIELD_THRESHOLD",
    "REFLECT_OVERALL_THRESHOLD",
    "REFLECT_LOG_LEVEL",
]
# Measured on the pre-iter-97 product (same digest the iter-96 ship pinned).
HEAD_FROZEN_EVAL_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"
HEAD_CLI_LINES = 306
HEAD_VERSION_STDOUT = "reflect-mcp 0.1.0\n"
HEAD_BASE_DEPS = ["pydantic>=2.6"]
HEAD_OPTIONAL_DEPS = {
    "server": ["mcp>=1.28,<2"],
    "anthropic": ["anthropic>=0.39"],
    "openai": ["openai>=1.30"],
    "bedrock": ["boto3>=1.34"],
    "dev": ["pytest>=8.0", "pytest-cov>=5.0"],
}


def _hermetic_env(**extra: str) -> dict[str, str]:
    """Inherited env minus every REFLECT_*, *_API_KEY and AWS_* var, plus extras."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("REFLECT_") or k.endswith("_API_KEY") or k.startswith("AWS_"))
    }
    env.update(extra)
    return env


def _run(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *argv],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=_hermetic_env() if env is None else env,
        cwd=str(REPO_ROOT),
        timeout=60,
    )


def _block(env: dict[str, str] | None = None) -> dict:
    r = _run(["--mcp-config"], env)
    assert r.returncode == 0, (r.returncode, r.stderr)
    return json.loads(r.stdout)


def _server_env(env: dict[str, str] | None = None) -> dict:
    return _block(env)["mcpServers"]["reflect-mcp"]["env"]


# ---------------------------------------------------------------- EB1
def test_eb1_valid_json_exact_shape_exit0_empty_stderr():
    r = _run(["--mcp-config"])
    assert r.returncode == 0
    assert r.stderr == ""
    doc = json.loads(r.stdout)
    assert isinstance(doc, dict)
    assert list(doc) == ["mcpServers"]
    assert list(doc["mcpServers"]) == ["reflect-mcp"]
    assert set(doc["mcpServers"]["reflect-mcp"]) == {"command", "args", "env"}


# ---------------------------------------------------------------- EB2
def test_eb2_command_is_this_interpreter_and_args_are_server_module():
    srv = _block()["mcpServers"]["reflect-mcp"]
    assert srv["command"] == sys.executable
    assert srv["args"] == ["-m", "reflect_mcp.server"]


def test_eb2_printed_command_can_import_package_without_extras():
    srv = _block()["mcpServers"]["reflect-mcp"]
    r = subprocess.run(
        [srv["command"], "-c", "import reflect_mcp, reflect_mcp.__main__"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=_hermetic_env(), cwd=str(REPO_ROOT), timeout=60,
    )
    assert r.returncode == 0, r.stderr


def test_eb2_printed_command_can_import_server_module():
    pytest.importorskip("mcp")
    srv = _block()["mcpServers"]["reflect-mcp"]
    r = subprocess.run(
        [srv["command"], "-c", "import reflect_mcp.server"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=_hermetic_env(), cwd=str(REPO_ROOT), timeout=60,
    )
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- EB3
def test_eb3_default_env_is_anthropic_with_placeholder_in_order():
    env = _server_env()
    assert env == {"REFLECT_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": PLACEHOLDER}
    assert list(env) == ["REFLECT_PROVIDER", "ANTHROPIC_API_KEY"]


# ---------------------------------------------------------------- EB4
def test_eb4_openai_gets_openai_key_placeholder_only():
    env = _server_env(_hermetic_env(REFLECT_PROVIDER="openai"))
    assert env["REFLECT_PROVIDER"] == "openai"
    assert env["OPENAI_API_KEY"] == PLACEHOLDER
    assert "ANTHROPIC_API_KEY" not in env
    assert list(env) == ["REFLECT_PROVIDER", "OPENAI_API_KEY"]


def test_eb4_bedrock_lowercased_and_has_no_api_key_var():
    env = _server_env(_hermetic_env(REFLECT_PROVIDER="Bedrock"))
    assert env["REFLECT_PROVIDER"] == "bedrock"
    assert [k for k in env if k.endswith("_API_KEY")] == []
    assert env == {"REFLECT_PROVIDER": "bedrock"}


def test_eb4_every_registered_provider_is_accepted():
    # AC: the key-var table must cover every provider the product can build.
    from reflect_mcp import providers

    names = sorted(providers._CLIENTS)
    assert names, "no providers registered"
    for name in names:
        env = _server_env(_hermetic_env(REFLECT_PROVIDER=name))
        assert env["REFLECT_PROVIDER"] == name
        assert all(v == PLACEHOLDER for k, v in env.items() if k.endswith("_API_KEY"))
    assert set(names) >= {"anthropic", "openai", "bedrock"}


# ---------------------------------------------------------------- EB5
@pytest.mark.parametrize("provider", ["anthropic", "openai", "bedrock"])
def test_eb5_never_echoes_a_secret(provider):
    secrets = {
        "ANTHROPIC_API_KEY": "sk-test-" + "SECRET-1",
        "OPENAI_API_KEY": "sk-test-" + "SECRET-2",
        "AWS_SECRET_ACCESS_KEY": "SECRET-3",
    }
    r = _run(["--mcp-config"], _hermetic_env(REFLECT_PROVIDER=provider, **secrets))
    assert r.returncode == 0, r.stderr
    for out in (r.stdout, r.stderr):
        assert "SECRET" not in out
        assert "sk-test" not in out
    assert json.loads(r.stdout)["mcpServers"]["reflect-mcp"]["env"]["REFLECT_PROVIDER"] == provider


# ---------------------------------------------------------------- EB6
ECHO_VALUES = {
    "REFLECT_GENERATOR_MODEL": "gen-x",
    "REFLECT_CRITIC_MODEL": "crit-y",
    "REFLECT_FIELD_THRESHOLD": "0.8",
    "REFLECT_OVERALL_THRESHOLD": "0.9",
    "REFLECT_LOG_LEVEL": "DEBUG",
}


def test_eb6_set_reflect_vars_echoed_verbatim_in_spec_order():
    # Pass the vars in REVERSE order so the output order cannot just mirror os.environ.
    extra = dict(reversed(list(ECHO_VALUES.items())))
    env = _server_env(_hermetic_env(**extra))
    for k, v in ECHO_VALUES.items():
        assert env[k] == v
    assert list(env) == ECHO_ORDER + ["ANTHROPIC_API_KEY"]
    assert env["ANTHROPIC_API_KEY"] == PLACEHOLDER


def test_eb6_unrelated_vars_not_echoed():
    env = _server_env(_hermetic_env(REFLECT_UNRELATED="z", AWS_REGION="eu-west-1",
                                    REFLECT_CRITIC_MODEL="crit-y"))
    assert "REFLECT_UNRELATED" not in env
    assert "AWS_REGION" not in env
    assert "z" not in env.values() and "eu-west-1" not in env.values()
    assert list(env) == ["REFLECT_PROVIDER", "REFLECT_CRITIC_MODEL", "ANTHROPIC_API_KEY"]


def test_eb6_empty_string_var_omitted():
    env = _server_env(_hermetic_env(REFLECT_GENERATOR_MODEL="", REFLECT_LOG_LEVEL="INFO"))
    assert "REFLECT_GENERATOR_MODEL" not in env
    assert env["REFLECT_LOG_LEVEL"] == "INFO"


# ---------------------------------------------------------------- EB7
def test_eb7_unknown_provider_is_usage_error():
    r = _run(["--mcp-config"], _hermetic_env(REFLECT_PROVIDER="bogus"))
    assert r.returncode == 2
    assert r.stdout == ""
    for token in ("bogus", "anthropic", "openai", "bedrock"):
        assert token in r.stderr, (token, r.stderr)


# ---------------------------------------------------------------- EB8
def test_eb8_help_lists_flag():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "--mcp-config" in r.stdout


def test_eb8_version_unchanged():
    r = _run(["--version"])
    assert r.returncode == 0
    assert r.stdout == HEAD_VERSION_STDOUT


def test_eb8_flag_with_verb_prints_block_and_runs_no_verb():
    base = _run(["--mcp-config"])
    # stdin carries a document; if the verb ran, it would consume it and print a result.
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "--mcp-config", "extract"],
        input="Invoice INV-1 total 10.00 USD", capture_output=True, text=True,
        env=_hermetic_env(), cwd=str(REPO_ROOT), timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == base.stdout
    assert '"status"' not in r.stdout
    assert r.stderr == ""


# ---------------------------------------------------------------- EB9
def test_eb9_cli_py_line_count_unchanged():
    cli = REPO_ROOT / "src" / "reflect_mcp" / "cli.py"
    assert len(cli.read_text(encoding="utf-8").splitlines()) == HEAD_CLI_LINES


def test_eb9_no_new_dependency():
    # Base and optional requirement lists measured on the pre-iter-97 pyproject.toml.
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
    assert m, "no dependencies list"
    assert re.findall(r'"([^"]+)"', m.group(1)) == HEAD_BASE_DEPS
    opt = re.search(r"^\[project\.optional-dependencies\]\n(.*?)^\[", text, re.S | re.M)
    assert opt, "no optional-dependencies table"
    found = {k: re.findall(r'"([^"]+)"', v)
             for k, v in re.findall(r"^(\w+)\s*=\s*\[(.*?)\]", opt.group(1), re.M)}
    assert found == HEAD_OPTIONAL_DEPS


def test_eb9_frozen_eval_stdout_byte_identical_to_head():
    proc = frozen_eval()
    assert proc.returncode == 0
    assert hashlib.sha256(proc.stdout.encode()).hexdigest() == HEAD_FROZEN_EVAL_SHA256


# ---------------------------------------------------------------- EB10
def _section4() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## 4\..*?(?=^## 5\.)", readme, re.S | re.M)
    assert m, "README section 4 not found"
    return m.group(0)


def test_eb10_readme_section4_mentions_mcp_config():
    s4 = _section4()
    assert "--mcp-config" in s4
    assert "reflect-mcp --mcp-config" in s4 or "python -m reflect_mcp --mcp-config" in s4


def test_eb10_readme_publication_safe():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert ("/" + "Users/") not in readme
    assert ("/" + "home/") not in readme
    for m in re.finditer("sk-" + "ant-" + r"\S*", readme):
        tail = m.group(0)
        assert "..." in tail or "<" in tail or "x" * 4 in tail.lower(), tail
