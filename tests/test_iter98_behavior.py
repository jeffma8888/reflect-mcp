"""Black-box behavioral tests for iter-98: re-land of ``reflect-mcp --mcp-config``
(roadmap #97) with the clean-install (no optional extras) fix.

Complements ``tests/test_iter97_behavior.py`` from independent angles:
  1  exact JSON shape, exit 0, empty stderr, a single JSON document on stdout
  2  printed command is this absolute interpreter; extras-free import works;
     the server-module import is skip-guarded on the optional ``mcp`` extra
  3  default env exact and ordered; explicit ``anthropic`` gives the same block
  4  per-provider key placeholder for every registered provider, case-folded
  5  exported fake secrets never surface, for every provider
  6  echo order is the spec order regardless of environment insertion order;
     unrelated / AWS / empty vars are omitted; values are verbatim
  7  unknown provider -> exit 2, empty stdout, stderr names it and valid ones
  8  --help lists the flag; --version tracks pyproject; the flag with a verb
     prints the same block and never reads stdin (structural: an open pipe)
  9  frozen eval keeps rc 0; no new runtime dependency
 10  README section 4 tells the user to run the flag and paste the output,
     with no machine path and no non-placeholder key
 11  the flag works with the ``mcp`` SDK made unimportable (clean-install shape)

Offline, fake keys only, no git at test time, no wall-clock bound.

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

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
    "REFLECT_GENERATOR_MODEL",
    "REFLECT_CRITIC_MODEL",
    "REFLECT_FIELD_THRESHOLD",
    "REFLECT_OVERALL_THRESHOLD",
    "REFLECT_LOG_LEVEL",
]
EXPECTED_KEY = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "bedrock": None}


def _hermetic_env(*pairs: tuple[str, str]) -> dict[str, str]:
    """Inherited env minus REFLECT_*, *_API_KEY and AWS_*; then pairs in order."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("REFLECT_") or k.endswith("_API_KEY") or k.startswith("AWS_"))
    }
    for k, v in pairs:
        env[k] = v
    return env


def _run(argv, env=None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *argv],
        capture_output=True,
        text=True,
        stdin=kw.pop("stdin", subprocess.DEVNULL),
        env=_hermetic_env() if env is None else env,
        cwd=str(REPO_ROOT),
        timeout=60,
        **kw,
    )


def _server(env=None) -> dict:
    r = _run(["--mcp-config"], env)
    assert r.returncode == 0, (r.returncode, r.stderr)
    return json.loads(r.stdout)["mcpServers"]["reflect-mcp"]


# ---------------------------------------------------------------- EB1
def test_eb1_single_json_document_exact_shape():
    r = _run(["--mcp-config"])
    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    # raw_decode consumes the whole stdout: one JSON document, nothing else.
    doc, end = json.JSONDecoder().raw_decode(r.stdout.lstrip())
    assert r.stdout.lstrip()[end:].strip() == ""
    assert list(doc) == ["mcpServers"]
    assert list(doc["mcpServers"]) == ["reflect-mcp"]
    srv = doc["mcpServers"]["reflect-mcp"]
    assert sorted(srv) == ["args", "command", "env"]
    assert isinstance(srv["command"], str)
    assert isinstance(srv["args"], list)
    assert isinstance(srv["env"], dict)
    assert all(isinstance(v, str) for v in srv["env"].values())


# ---------------------------------------------------------------- EB2
def test_eb2_command_is_absolute_existing_interpreter():
    srv = _server()
    assert srv["command"] == sys.executable
    assert os.path.isabs(srv["command"])
    assert os.path.exists(srv["command"])
    assert srv["args"] == ["-m", "reflect_mcp.server"]


def test_eb2_printed_command_imports_package_without_extras_from_foreign_cwd(tmp_path):
    # Hosts launch the command from an arbitrary cwd, not the repo root.
    srv = _server()
    r = subprocess.run(
        [srv["command"], "-c", "import reflect_mcp, reflect_mcp.__main__; print('ok')"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=_hermetic_env(), cwd=str(tmp_path), timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_eb2_printed_command_and_args_import_server_when_sdk_present(tmp_path):
    pytest.importorskip("mcp")
    srv = _server()
    r = subprocess.run(
        [srv["command"], "-c", "import importlib.util as u, sys; "
         f"sys.exit(0 if u.find_spec({srv['args'][1]!r}) else 3)"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=_hermetic_env(), cwd=str(tmp_path), timeout=60,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [srv["command"], "-c", "import reflect_mcp.server"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=_hermetic_env(), cwd=str(tmp_path), timeout=60,
    )
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- EB3
def test_eb3_default_env_exact_and_ordered():
    env = _server()["env"]
    assert list(env.items()) == [
        ("REFLECT_PROVIDER", "anthropic"),
        ("ANTHROPIC_API_KEY", PLACEHOLDER),
    ]


def test_eb3_explicit_anthropic_equals_default():
    assert _server(_hermetic_env(("REFLECT_PROVIDER", "anthropic"))) == _server()


# ---------------------------------------------------------------- EB4
@pytest.mark.parametrize("raw", ["openai", "OpenAI", "OPENAI"])
def test_eb4_openai_placeholder_case_folded(raw):
    env = _server(_hermetic_env(("REFLECT_PROVIDER", raw)))["env"]
    assert list(env.items()) == [("REFLECT_PROVIDER", "openai"), ("OPENAI_API_KEY", PLACEHOLDER)]


@pytest.mark.parametrize("raw", ["bedrock", "Bedrock", "BEDROCK"])
def test_eb4_bedrock_has_no_api_key_var(raw):
    env = _server(_hermetic_env(("REFLECT_PROVIDER", raw)))["env"]
    assert env == {"REFLECT_PROVIDER": "bedrock"}
    assert not any(k.endswith("_API_KEY") for k in env)


def test_eb4_every_registered_provider_gets_its_placeholder():
    from reflect_mcp import providers

    registered = set(providers._CLIENTS)
    assert registered == set(EXPECTED_KEY)
    for name in sorted(registered):
        env = _server(_hermetic_env(("REFLECT_PROVIDER", name)))["env"]
        assert env["REFLECT_PROVIDER"] == name
        keys = [k for k in env if k.endswith("_API_KEY")]
        want = EXPECTED_KEY[name]
        assert keys == ([want] if want else []), (name, env)
        if want:
            assert env[want] == PLACEHOLDER
            assert list(env)[-1] == want


# ---------------------------------------------------------------- EB5
@pytest.mark.parametrize("provider", ["anthropic", "openai", "bedrock"])
def test_eb5_exported_secrets_never_surface(provider):
    marker = "S" + "ECRET"
    env = _hermetic_env(
        ("REFLECT_PROVIDER", provider),
        ("ANTHROPIC_API_KEY", "sk-test-" + marker + "-a1"),
        ("OPENAI_API_KEY", "sk-test-" + marker + "-o2"),
        ("AWS_ACCESS_KEY_ID", "AKIA" + marker + "X"),
        ("AWS_SECRET_ACCESS_KEY", marker + "-aws3"),
        ("AWS_SESSION_TOKEN", marker + "-tok4"),
    )
    r = _run(["--mcp-config"], env)
    assert r.returncode == 0, r.stderr
    for stream in (r.stdout, r.stderr):
        assert marker not in stream
        assert "sk-test" not in stream
        assert "AKIA" not in stream
    srv = json.loads(r.stdout)["mcpServers"]["reflect-mcp"]
    for k, v in srv["env"].items():
        if k.endswith("_API_KEY"):
            assert v == PLACEHOLDER


# ---------------------------------------------------------------- EB6
VALUES = {
    "REFLECT_GENERATOR_MODEL": "gen-model x/1",
    "REFLECT_CRITIC_MODEL": "critic:model-ü",
    "REFLECT_FIELD_THRESHOLD": "0.81",
    "REFLECT_OVERALL_THRESHOLD": "0.9",
    "REFLECT_LOG_LEVEL": "DEBUG",
}


@pytest.mark.parametrize("insertion", [ECHO_ORDER, list(reversed(ECHO_ORDER)),
                                       ECHO_ORDER[2:] + ECHO_ORDER[:2]])
def test_eb6_echo_order_independent_of_insertion_order(insertion):
    pairs = [(k, VALUES[k]) for k in insertion]
    # provider set last, after all echo vars, to prove it is still emitted first
    env = _server(_hermetic_env(*pairs, ("REFLECT_PROVIDER", "openai")))["env"]
    assert list(env) == ["REFLECT_PROVIDER", *ECHO_ORDER, "OPENAI_API_KEY"]
    for k in ECHO_ORDER:
        assert env[k] == VALUES[k]


def test_eb6_unrelated_aws_and_empty_vars_omitted():
    env = _server(_hermetic_env(
        ("REFLECT_UNRELATED", "nope"),
        ("REFLECT_HOME", "nope"),
        ("AWS_REGION", "us-east-1"),
        ("AWS_PROFILE", "dev"),
        ("REFLECT_CRITIC_MODEL", ""),
        ("REFLECT_LOG_LEVEL", "INFO"),
    ))["env"]
    assert list(env.items()) == [
        ("REFLECT_PROVIDER", "anthropic"),
        ("REFLECT_LOG_LEVEL", "INFO"),
        ("ANTHROPIC_API_KEY", PLACEHOLDER),
    ]


def test_eb6_subset_keeps_relative_order():
    env = _server(_hermetic_env(
        ("REFLECT_LOG_LEVEL", "WARNING"),
        ("REFLECT_GENERATOR_MODEL", "g"),
    ))["env"]
    assert list(env) == ["REFLECT_PROVIDER", "REFLECT_GENERATOR_MODEL",
                         "REFLECT_LOG_LEVEL", "ANTHROPIC_API_KEY"]


# ---------------------------------------------------------------- EB7
@pytest.mark.parametrize("bad", ["bogus", "gemini"])
def test_eb7_unknown_provider_usage_error(bad):
    r = _run(["--mcp-config"], _hermetic_env(("REFLECT_PROVIDER", bad)))
    assert r.returncode == 2
    assert r.stdout == ""
    for token in (bad, "anthropic", "openai", "bedrock"):
        assert token in r.stderr, (token, r.stderr)


# ---------------------------------------------------------------- EB8
def test_eb8_help_lists_flag():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "--mcp-config" in r.stdout


def test_eb8_version_tracks_pyproject():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m
    r = _run(["--version"])
    assert r.returncode == 0
    assert r.stdout == f"reflect-mcp {m.group(1)}\n"


def test_eb8_flag_with_verb_same_block_and_never_reads_stdin():
    baseline = _run(["--mcp-config"])
    assert baseline.returncode == 0
    # An open, never-written, never-closed stdin pipe: any stdin read would
    # block until the timeout kills the child (structural, not a timing ratio).
    proc = subprocess.Popen(
        [sys.executable, "-m", "reflect_mcp", "--mcp-config", "extract"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=_hermetic_env(), cwd=str(REPO_ROOT),
    )
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail("--mcp-config extract blocked on stdin")
    out = proc.stdout.read()
    proc.stdin.close()
    proc.stdout.close()
    proc.stderr.close()
    assert rc == 0
    assert json.loads(out) == json.loads(baseline.stdout)


# ---------------------------------------------------------------- EB9
def test_eb9_frozen_eval_still_green():
    proc = frozen_eval()
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip()


def test_eb9_base_dependencies_unchanged():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
    assert m
    assert re.findall(r'"([^"]+)"', m.group(1)) == ["pydantic>=2.6"]


# ---------------------------------------------------------------- EB10
def _section4() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## 4\..*?(?=^## 5\.)", readme, re.S | re.M)
    assert m, "README section 4 not found"
    return m.group(0)


def test_eb10_section4_says_run_flag_and_paste():
    s4 = _section4()
    assert "reflect-mcp --mcp-config" in s4
    assert "paste" in s4.lower()
    assert PLACEHOLDER in s4


def test_eb10_readme_publication_safe():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for token in ("/" + "Users/", "/" + "home/", "C:" + "\\Users"):
        assert token not in readme, token
    key_prefix = "sk-" + "ant-"
    for hit in re.finditer(re.escape(key_prefix) + r"\S*", readme):
        tail = hit.group(0)[len(key_prefix):]
        assert tail.startswith(("...", "<", "xxx", "XXX")) or "your" in tail.lower(), hit.group(0)


# ---------------------------------------------------------------- EB11
def test_eb11_flag_works_with_mcp_sdk_unimportable():
    # Clean-install shape: the optional server extra is absent. A None entry in
    # sys.modules makes any `import mcp` raise ImportError.
    code = (
        "import sys, runpy; sys.modules['mcp'] = None; "
        "sys.argv = ['reflect-mcp', '--mcp-config']; "
        "runpy.run_module('reflect_mcp', run_name='__main__', alter_sys=True)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        stdin=subprocess.DEVNULL, env=_hermetic_env(), cwd=str(REPO_ROOT), timeout=60,
    )
    assert r.returncode == 0, r.stderr
    srv = json.loads(r.stdout)["mcpServers"]["reflect-mcp"]
    assert srv["args"] == ["-m", "reflect_mcp.server"]
    assert srv["env"]["REFLECT_PROVIDER"] == "anthropic"
