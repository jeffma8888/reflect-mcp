"""Black-box behavioral tests for iter-85: MCP ``serverInfo.version`` is ours.

Covers the Expected Behaviors from pm.md:
  1. Real SDK, in-process: ``mcp._mcp_server.create_initialization_options()``
     reports ``server_version == reflect_mcp.__version__`` and name
     ``"reflect-mcp"`` (importorskip("mcp")). Red on HEAD 9decb18 (1.28.1).
  2. Real SDK, live stdio handshake: one JSON-RPC ``initialize`` line in, one
     reply line out; ``result.serverInfo`` name/version are ours; the process
     exits on its own after stdin closes (importorskip("mcp")). Red on HEAD.
  3. Stub path stays importable: a FastMCP stub WITHOUT ``_mcp_server`` still
     imports ``reflect_mcp.server`` with no exception, no log record and no
     stderr output.
  4. Version single-sourcing pin (drift guard, green on HEAD by design):
     pyproject ``[project].version`` (read by regex; no tomllib on 3.10)
     == ``reflect_mcp.__version__`` == ``reflect-mcp --version``.
  5. Nothing else moves: tools/list (name, inputSchema, outputSchema) digest
     equals HEAD 9decb18's; frozen-subset eval stdout digest unchanged;
     base dependencies and the ``server`` extra unchanged; non-server modules
     import with ``mcp`` blocked.
  README: one line ties the host-visible ``serverInfo.version`` to ``--version``.

Isolation contract: only the public surface is driven (module import, the
SDK's own initialization options / list_tools, the stdio server process and
the CLI). HEAD digests were measured on a ``git archive 9decb18`` copy run
with ``PYTHONPATH=<copy>/src <repo>/.venv/bin/python``.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

import reflect_mcp
from tests._golden import FROZEN_EVAL_SHA256, frozen_eval_sha256

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
README = REPO / "README.md"

# sha256 of json.dumps([{name, inputSchema, outputSchema} sorted by name],
# sort_keys=True) measured on HEAD 9decb18 with the real mcp 1.28.1 SDK.
HEAD_TOOLS_LIST_SHA256 = "41fb3cc9bfe6c3877bfef150175769600a444c6e7ce0e9d45a3d7124582b195d"
# [project] dependencies and the `server` extra as of HEAD 9decb18.
HEAD_BASE_DEPENDENCIES = ["pydantic>=2.6"]
HEAD_SERVER_EXTRA = ["mcp>=1.28,<2"]

HANDSHAKE_TIMEOUT = 10.0


def _pyproject_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert m, "no [project] table in pyproject.toml"
    v = re.search(r'^version\s*=\s*"([^"]+)"', m.group(1), re.M)
    assert v, "no version in [project]"
    return v.group(1)


@pytest.fixture
def real_server():
    pytest.importorskip("mcp")
    pytest.importorskip("mcp.server.fastmcp")
    sys.modules.pop("reflect_mcp.server", None)
    try:
        yield importlib.import_module("reflect_mcp.server")
    finally:
        sys.modules.pop("reflect_mcp.server", None)


# ---------------------------------------------------------------------------
# EB1: real SDK, in-process initialization options
# ---------------------------------------------------------------------------

class TestEB1InitOptions:
    def test_server_version_is_package_version(self, real_server):
        opts = real_server.mcp._mcp_server.create_initialization_options()
        assert opts.server_version == reflect_mcp.__version__

    def test_server_version_is_not_sdk_version(self, real_server):
        from importlib.metadata import version

        sdk = version("mcp")
        if sdk == reflect_mcp.__version__:  # pragma: no cover - coincidence guard
            pytest.skip("SDK and package versions coincide; not discriminating")
        opts = real_server.mcp._mcp_server.create_initialization_options()
        assert opts.server_version != sdk

    def test_server_name_unchanged(self, real_server):
        opts = real_server.mcp._mcp_server.create_initialization_options()
        assert opts.server_name == "reflect-mcp"


# ---------------------------------------------------------------------------
# EB2: real SDK, live stdio handshake
# ---------------------------------------------------------------------------

def _clean_env() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    # Keep the import path the suite runs under (src layout / editable install),
    # but no provider credentials or REFLECT_* configuration.
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return env


def _handshake() -> tuple[dict, int | None]:
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"},
        },
    }
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.Popen(
            [sys.executable, "-m", "reflect_mcp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=cwd,
            env=_clean_env(),
            text=True,
        )
        try:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            box: list[str] = []

            import threading

            t = threading.Thread(target=lambda: box.append(proc.stdout.readline()), daemon=True)
            t.start()
            t.join(HANDSHAKE_TIMEOUT)
            assert box and box[0].strip(), "no initialize reply within timeout"
            reply = json.loads(box[0])
            proc.stdin.close()
            try:
                rc = proc.wait(timeout=HANDSHAKE_TIMEOUT)
            except subprocess.TimeoutExpired:
                rc = None
            return reply, rc
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.fixture(scope="module")
def handshake():
    pytest.importorskip("mcp")
    return _handshake()


class TestEB2StdioHandshake:
    def test_reply_is_jsonrpc_result_for_id_1(self, handshake):
        reply, _ = handshake
        assert reply.get("jsonrpc") == "2.0"
        assert reply.get("id") == 1
        assert "result" in reply and "error" not in reply, reply

    def test_server_info_name_and_version(self, handshake):
        reply, _ = handshake
        info = reply["result"]["serverInfo"]
        assert {"name": info.get("name"), "version": info.get("version")} == {
            "name": "reflect-mcp",
            "version": reflect_mcp.__version__,
        }, info

    def test_protocol_version_echoed(self, handshake):
        reply, _ = handshake
        assert reply["result"]["protocolVersion"] == "2025-06-18"

    def test_exits_on_its_own_after_stdin_closes(self, handshake):
        _, rc = handshake
        assert rc is not None, "server did not exit within timeout after stdin closed"
        assert rc == 0


# ---------------------------------------------------------------------------
# EB3: stub FastMCP without _mcp_server still imports silently
# ---------------------------------------------------------------------------

class _BareFastMCP:
    """FastMCP stand-in with no ``_mcp_server`` (shape of the iter-74 stub)."""

    instances: list = []

    def __init__(self, name, *args, **kwargs):
        self.name = name
        _BareFastMCP.instances.append(self)

    def tool(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    def run(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("stub run must not be called")


class _Any:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def stub_import(monkeypatch, caplog, capsys):
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    types_mod = types.ModuleType("mcp.types")
    fastmcp_mod.FastMCP = _BareFastMCP
    types_mod.CallToolResult = _Any
    types_mod.TextContent = _Any
    mcp_mod.server = server_mod
    mcp_mod.types = types_mod
    server_mod.fastmcp = fastmcp_mod
    for name, mod in [("mcp", mcp_mod), ("mcp.server", server_mod),
                      ("mcp.server.fastmcp", fastmcp_mod), ("mcp.types", types_mod)]:
        monkeypatch.setitem(sys.modules, name, mod)
    _BareFastMCP.instances.clear()
    sys.modules.pop("reflect_mcp.server", None)
    capsys.readouterr()
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        try:
            mod = importlib.import_module("reflect_mcp.server")
            captured = capsys.readouterr()
            records = list(caplog.records)
            yield mod, captured, records
        finally:
            sys.modules.pop("reflect_mcp.server", None)


class TestEB3StubPath:
    def test_imports_with_bare_stub(self, stub_import):
        mod, _, _ = stub_import
        assert isinstance(mod.mcp, _BareFastMCP)
        assert mod.mcp.name == "reflect-mcp"
        assert callable(mod.extract) and callable(mod.validate)

    def test_stub_still_has_no_mcp_server(self, stub_import):
        mod, _, _ = stub_import
        # The guard must not fabricate the attribute on a stub that lacks it.
        assert not hasattr(mod.mcp, "_mcp_server")

    def test_no_log_records_and_no_stderr(self, stub_import):
        _, captured, records = stub_import
        assert records == [], [r.getMessage() for r in records]
        assert captured.err == ""
        assert captured.out == ""


# ---------------------------------------------------------------------------
# EB4: version single-sourcing pin (drift guard, green on HEAD by design)
# ---------------------------------------------------------------------------

class TestEB4VersionPin:
    def test_pyproject_equals_dunder_version(self):
        assert _pyproject_version() == reflect_mcp.__version__

    def test_cli_version_prints_same_value(self):
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                [sys.executable, "-m", "reflect_mcp", "--version"],
                capture_output=True, text=True, cwd=cwd, env=_clean_env(), timeout=60,
            )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == f"reflect-mcp {_pyproject_version()}"

    def test_regex_reads_project_table_not_other_tables(self, tmp_path):
        # The pin's own reader must take [project].version, not a later table's.
        fake = tmp_path / "pyproject.toml"
        fake.write_text('[tool.x]\nversion = "9.9.9"\n[project]\nname = "a"\nversion = "1.2.3"\n'
                        '[tool.y]\nversion = "7.7.7"\n', encoding="utf-8")
        text = fake.read_text(encoding="utf-8")
        m = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
        v = re.search(r'^version\s*=\s*"([^"]+)"', m.group(1), re.M)
        assert v.group(1) == "1.2.3"


# ---------------------------------------------------------------------------
# EB5: nothing else moves
# ---------------------------------------------------------------------------

def _toml_list(table: str, key: str) -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(rf"^\[{re.escape(table)}\]\s*$(.*?)(?=^\[[^\]\"']|\Z)", text, re.M | re.S)
    assert m, table
    k = re.search(rf"^{re.escape(key)}\s*=\s*\[(.*?)\]\s*$", m.group(1), re.M | re.S)
    assert k, (table, key)
    return re.findall(r'"([^"]+)"', k.group(1))


class TestEB5NothingElseMoves:
    def test_tools_list_digest_unchanged(self, real_server):
        listed = asyncio.run(real_server.mcp.list_tools())
        blob = json.dumps(
            [{"name": t.name, "inputSchema": t.inputSchema, "outputSchema": t.outputSchema}
             for t in sorted(listed, key=lambda t: t.name)],
            sort_keys=True,
        )
        assert hashlib.sha256(blob.encode()).hexdigest() == HEAD_TOOLS_LIST_SHA256

    def test_frozen_eval_stdout_unchanged(self):
        assert frozen_eval_sha256() == FROZEN_EVAL_SHA256

    def test_no_new_dependency(self):
        assert _toml_list("project", "dependencies") == HEAD_BASE_DEPENDENCIES
        assert _toml_list("project.optional-dependencies", "server") == HEAD_SERVER_EXTRA

    def test_non_server_modules_import_without_mcp(self):
        code = (
            "import sys, pkgutil, importlib\n"
            "for n in ['mcp','mcp.server','mcp.server.fastmcp','mcp.types']: sys.modules[n]=None\n"
            "import reflect_mcp\n"
            "names=[m.name for m in pkgutil.iter_modules(reflect_mcp.__path__)"
            " if m.name not in {'server','__main__'}]\n"
            "assert names, 'no submodules found'\n"
            "for n in names: importlib.import_module('reflect_mcp.'+n)\n"
            "print('OK', len(names))\n"
        )
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                  cwd=cwd, env=_clean_env(), timeout=60)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.startswith("OK ")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

class TestReadme:
    def test_readme_ties_server_info_to_cli_version(self):
        text = README.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines()
                 if "serverInfo.version" in ln and "--version" in ln]
        assert lines, "README has no line tying serverInfo.version to --version"
