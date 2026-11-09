"""Black-box behavioral tests for iter-79: ``validate`` rejects caller ``data``
whose keys are not ``InvoiceData`` fields (a typo such as ``vendr_name`` used to
be silently dropped and verify as ``ok``).

Covers the 9 Expected Behaviors from pm.md:
  1. CLI --data with one unknown key -> exit 2, empty stdout, stderr names it
     and lists the known fields in canonical order.
  2. Several unknown keys -> sorted, ", "-joined; known keys never listed.
  3. Known-only data unchanged vs HEAD (incl. explicit null and ``{}``).
  4. Stored envelopes: only the inner ``data`` is checked.
  5. Non-envelope dicts with stray keys (``status``/``data`` beside a known
     field) are rejected.
  6. --batch: the unknown-key row becomes an id-led error row, exit 3, same
     output at --batch-concurrency 1 and 4.
  7. MCP validate_tool returns status:"error" (never raises); the server tool
     path sets isError; known-only data unchanged vs HEAD.
  8. Generator replies keep ignoring extra keys; context-free
     ``InvoiceData.model_validate`` still ignores extras.
  9. ``eval/run_eval.py`` stdout + exit code identical to HEAD; cli.py keeps
     its 375-line count.

Isolation contract: tests run the product as a subprocess
(``python -m reflect_mcp``) or call its public Python API with scripted LLM
clients, and assert observable output only. No implementation source was read.

"HEAD" is the iter-78 commit (``BASE_SHA``). Its outputs are produced live by
running the product from a ``git archive BASE_SHA`` tree
(``PYTHONPATH=<tree>/src``); those comparisons skip when the commit is not
reachable (e.g. on an archive copy), while every absolute assertion still runs.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
import tarfile
import io
import types
from pathlib import Path

import pytest

from reflect_mcp import tools
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import critique_all

REPO = Path(__file__).resolve().parents[1]
BASE_SHA = "7be732b"  # iter-78 ship, the HEAD this iteration's spec measured

SRC = "Invoice #: INV-1\nTotal: 10.00"
KNOWN = "vendor_name, invoice_number, invoice_date, subtotal, tax, total, currency"
VENDR = {"total": 10.0, "vendr_name": "Acme"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(tree: Path) -> dict:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(tree / "src"),
        "HOME": os.environ.get("HOME", "/tmp"),
    }


def _cli(args: list[str], stdin: str = SRC, tree: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *args],
        input=stdin, capture_output=True, text=True, env=_env(tree),
        cwd=str(tree), timeout=120,
    )


def _validate(data, tree: Path = REPO) -> subprocess.CompletedProcess:
    payload = data if isinstance(data, str) else json.dumps(data)
    return _cli(["validate", "--heuristics-only", "--data", payload], tree=tree)


def _minus_trace(d: dict) -> dict:
    d = dict(d)
    d.pop("trace_id", None)
    return d


def _single(proc: subprocess.CompletedProcess) -> str:
    """Single-document stdout as canonical text minus trace_id (key order kept)."""
    return json.dumps(_minus_trace(json.loads(proc.stdout)))


def _unknown_list(stderr_or_msg: str) -> str:
    """The text between 'unknown field(s): ' and ' (known:'."""
    marker = "unknown field(s): "
    assert marker in stderr_or_msg, stderr_or_msg[-2000:]
    rest = stderr_or_msg.split(marker, 1)[1]
    assert " (known: " in rest, rest[:500]
    return rest.split(" (known: ", 1)[0]


@pytest.fixture(scope="module")
def head_tree(tmp_path_factory) -> Path:
    """Extract the iter-78 commit into a temp dir, or skip if unreachable."""
    try:
        blob = subprocess.run(
            ["git", "archive", BASE_SHA], cwd=str(REPO), capture_output=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if blob.returncode != 0 or not blob.stdout:
        pytest.skip(f"{BASE_SHA} not reachable from this checkout")
    dest = tmp_path_factory.mktemp("head78")
    with tarfile.open(fileobj=io.BytesIO(blob.stdout)) as tf:
        tf.extractall(dest)
    return dest


# ---------------------------------------------------------------------------
# EB1: single unknown key on the CLI
# ---------------------------------------------------------------------------

class TestEB1SingleUnknownKey:
    def test_exit_2_and_empty_stdout(self):
        p = _validate(VENDR)
        assert p.returncode == 2, (p.stdout, p.stderr[-2000:])
        assert p.stdout == ""

    def test_stderr_names_key_and_known_list(self):
        p = _validate(VENDR)
        assert "Error: --data failed InvoiceData validation" in p.stderr
        assert "unknown field(s): vendr_name" in p.stderr
        assert f"(known: {KNOWN})" in p.stderr
        assert _unknown_list(p.stderr) == "vendr_name"

    def test_no_traceback(self):
        assert "Traceback" not in _validate(VENDR).stderr

    def test_head_was_silent_ok(self, head_tree):
        # Documents the bug being fixed (the spec's HEAD column).
        p = _validate(VENDR, tree=head_tree)
        assert p.returncode == 0
        assert json.loads(p.stdout)["status"] == "ok"


# ---------------------------------------------------------------------------
# EB2: several unknown keys, sorted; known keys never listed
# ---------------------------------------------------------------------------

class TestEB2SeveralUnknownKeys:
    def test_sorted_comma_space_joined(self):
        p = _validate({"totl": 999, "invoice_no": "INV-9"})
        assert p.returncode == 2, p.stderr[-2000:]
        assert p.stdout == ""
        assert "unknown field(s): invoice_no, totl" in p.stderr
        assert _unknown_list(p.stderr) == "invoice_no, totl"

    @pytest.mark.parametrize("data, expected", [
        ({"total": 10.0, "vendor_name": "X", "totl": 1}, "totl"),
        ({"zeta": 1, "total": 10.0, "alpha": 2, "currency": "USD", "mid": 3}, "alpha, mid, zeta"),
        ({"invoice_number": "INV-1", "Total": 10.0}, "Total"),
    ])
    def test_known_keys_not_listed(self, data, expected):
        p = _validate(data)
        assert p.returncode == 2, p.stderr[-2000:]
        listed = _unknown_list(p.stderr)
        assert listed == expected
        known_present = [k for k in data if k in KNOWN.split(", ")]
        for k in known_present:
            assert k not in listed.split(", ")


# ---------------------------------------------------------------------------
# EB3: known-only data unchanged
# ---------------------------------------------------------------------------

KNOWN_ONLY = [
    {"total": 10.0},
    {"vendor_name": None, "total": 10.0},
    {},
]


class TestEB3KnownOnlyUnchanged:
    @pytest.mark.parametrize("data", KNOWN_ONLY, ids=["total", "explicit-null", "empty"])
    def test_ok_exit_0(self, data):
        p = _validate(data)
        assert p.returncode == 0, p.stderr[-2000:]
        assert json.loads(p.stdout)["status"] == "ok"

    @pytest.mark.parametrize("data", KNOWN_ONLY, ids=["total", "explicit-null", "empty"])
    def test_identical_to_head(self, data, head_tree):
        new, old = _validate(data), _validate(data, tree=head_tree)
        assert (new.returncode, _single(new)) == (old.returncode, _single(old))

    def test_empty_keeps_nothing_verified_note(self, head_tree):
        new = json.loads(_validate({}).stdout)["notes"]
        old = json.loads(_validate({}, tree=head_tree).stdout)["notes"]
        assert new == old


# ---------------------------------------------------------------------------
# EB4: stored envelopes -- only inner data is checked
# ---------------------------------------------------------------------------

ENV_OK = {"status": "ok", "data": {"total": 10.0}}
ENV_BAD = {"status": "ok", "data": {"total": 10.0, "vendr_name": "Acme"}}


class TestEB4StoredEnvelopes:
    def test_minimal_envelope_ok(self):
        p = _validate(ENV_OK)
        assert p.returncode == 0, p.stderr[-2000:]
        assert json.loads(p.stdout)["status"] == "ok"

    def test_minimal_envelope_identical_to_head(self, head_tree):
        new, old = _validate(ENV_OK), _validate(ENV_OK, tree=head_tree)
        assert (new.returncode, _single(new)) == (old.returncode, _single(old))

    def test_full_saved_row_fed_back(self):
        saved = _validate({"total": 10.0})
        assert saved.returncode == 0
        row = json.loads(saved.stdout)
        # The saved row carries envelope keys that are not InvoiceData fields.
        assert {"status", "data", "field_confidence", "trace_id", "notes"} <= set(row)
        p = _validate(saved.stdout)
        assert p.returncode == 0, p.stderr[-2000:]
        assert json.loads(p.stdout)["status"] == "ok"
        assert "unknown field" not in p.stderr

    def test_full_saved_row_identical_to_head(self, head_tree):
        saved = _validate({"total": 10.0}, tree=head_tree).stdout
        new, old = _validate(saved), _validate(saved, tree=head_tree)
        assert (new.returncode, _single(new)) == (old.returncode, _single(old))

    def test_envelope_with_bad_inner_key_rejected(self):
        p = _validate(ENV_BAD)
        assert p.returncode == 2, (p.stdout, p.stderr[-2000:])
        assert p.stdout == ""
        assert _unknown_list(p.stderr) == "vendr_name"
        listed = _unknown_list(p.stderr).split(", ")
        assert "status" not in listed and "data" not in listed


# ---------------------------------------------------------------------------
# EB5: non-envelope dicts with stray keys are rejected
# ---------------------------------------------------------------------------

class TestEB5NonEnvelopeStrayKeys:
    def test_status_only_rejected(self):
        p = _validate({"status": "paid"})
        assert p.returncode == 2, (p.stdout, p.stderr[-2000:])
        assert p.stdout == ""
        assert _unknown_list(p.stderr) == "status"

    def test_known_top_level_key_blocks_unwrap(self):
        p = _validate({"status": "ok", "vendor_name": "WRONG CO", "data": {"total": 10.0}})
        assert p.returncode == 2, (p.stdout, p.stderr[-2000:])
        assert p.stdout == ""
        assert "unknown field(s): data, status" in p.stderr
        assert _unknown_list(p.stderr) == "data, status"


# ---------------------------------------------------------------------------
# EB6: --batch row
# ---------------------------------------------------------------------------

BATCH = (
    json.dumps({"id": "a", "text": SRC, "data": {"total": 10.0}}) + "\n"
    + json.dumps({"id": "b", "text": SRC, "data": {"total": 10.0, "vendr_name": "Acme"}}) + "\n"
)


def _batch(conc: int, tree: Path = REPO) -> subprocess.CompletedProcess:
    return _cli(["validate", "--batch", "--heuristics-only", "--batch-concurrency", str(conc)],
                stdin=BATCH, tree=tree)


def _rows(proc) -> list[dict]:
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


class TestEB6BatchRow:
    @pytest.mark.parametrize("conc", [1, 4])
    def test_two_rows_exit_3(self, conc):
        p = _batch(conc)
        assert p.returncode == 3, (p.stdout, p.stderr[-2000:])
        rows = _rows(p)
        assert [r.get("id") for r in rows] == ["a", "b"]
        assert rows[0]["status"] == "ok"

    @pytest.mark.parametrize("conc", [1, 4])
    def test_error_row_shape(self, conc):
        b = _rows(_batch(conc))[1]
        assert list(b)[0] == "id"
        assert set(b) == {"id", "status", "error_message"}, b
        assert b["status"] == "error"
        assert "unknown field(s): vendr_name" in b["error_message"]
        assert _unknown_list(b["error_message"]) == "vendr_name"

    def test_concurrency_1_and_4_identical(self):
        one, four = _batch(1), _batch(4)
        assert one.returncode == four.returncode == 3
        strip = lambda p: [_minus_trace(r) for r in _rows(p)]  # noqa: E731
        assert strip(one) == strip(four)

    @pytest.mark.parametrize("conc", [1, 4])
    def test_row_a_identical_to_head(self, conc, head_tree):
        new = _rows(_batch(conc))[0]
        old = _rows(_batch(conc, tree=head_tree))[0]
        assert json.dumps(_minus_trace(new)) == json.dumps(_minus_trace(old))

    def test_head_had_both_ok(self, head_tree):
        p = _batch(1, tree=head_tree)
        assert p.returncode == 0
        assert [r["status"] for r in _rows(p)] == ["ok", "ok"]


# ---------------------------------------------------------------------------
# EB7: MCP tool
# ---------------------------------------------------------------------------

class _StubCallToolResult:
    def __init__(self, content=None, structuredContent=None, isError=False, **kwargs):
        self.content = content
        self.structuredContent = structuredContent
        self.isError = isError


class _StubTextContent:
    def __init__(self, type, text, **kwargs):
        self.type = type
        self.text = text


class _StubFastMCP:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def tool(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    def run(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("stub FastMCP.run must not be called in tests")


@pytest.fixture
def server(monkeypatch):
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    types_mod = types.ModuleType("mcp.types")
    fastmcp_mod.FastMCP = _StubFastMCP
    types_mod.CallToolResult = _StubCallToolResult
    types_mod.TextContent = _StubTextContent
    mcp_mod.server = server_mod
    mcp_mod.types = types_mod
    server_mod.fastmcp = fastmcp_mod
    for name, mod in (("mcp", mcp_mod), ("mcp.server", server_mod),
                      ("mcp.server.fastmcp", fastmcp_mod), ("mcp.types", types_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    # Re-bind package attributes (another suite module re-imports submodules).
    import reflect_mcp
    for full, mod in list(sys.modules.items()):
        if full.startswith("reflect_mcp.") and full.count(".") == 1 and mod is not None:
            monkeypatch.setattr(reflect_mcp, full.split(".", 1)[1], mod, raising=False)
    sys.modules.pop("reflect_mcp.server", None)
    mod = importlib.import_module("reflect_mcp.server")
    try:
        yield mod
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _head_tool_json(head_tree: Path, data: dict) -> dict:
    code = (
        "import json, sys; from reflect_mcp import tools; "
        f"r = tools.validate_tool({SRC!r}, json.loads(sys.argv[1]), heuristics_only=True); "
        "print(json.dumps(r))"
    )
    p = subprocess.run([sys.executable, "-c", code, json.dumps(data)], capture_output=True,
                       text=True, env=_env(head_tree), cwd=str(head_tree), timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


class TestEB7McpTool:
    def test_unknown_key_is_error_envelope(self):
        r = tools.validate_tool(SRC, dict(VENDR), heuristics_only=True)
        assert isinstance(r, dict)
        assert r["status"] == "error"
        assert "unknown field(s): vendr_name" in r["error_message"]
        assert _unknown_list(r["error_message"]) == "vendr_name"
        assert f"(known: {KNOWN})" in r["error_message"]

    def test_server_path_sets_is_error(self, server):
        res = asyncio.run(server.validate(SRC, dict(VENDR), heuristics_only=True))
        assert res.isError is True
        assert res.structuredContent["status"] == "error"
        assert "unknown field(s): vendr_name" in res.structuredContent["error_message"]
        assert json.loads(res.content[0].text) == res.structuredContent

    def test_server_known_only_not_error(self, server):
        res = asyncio.run(server.validate(SRC, {"total": 10.0}, heuristics_only=True))
        assert res.isError is False
        assert res.structuredContent["status"] == "ok"

    def test_known_only_ok(self):
        r = tools.validate_tool(SRC, {"total": 10.0}, heuristics_only=True)
        assert r["status"] == "ok"

    def test_known_only_identical_to_head(self, head_tree):
        new = tools.validate_tool(SRC, {"total": 10.0}, heuristics_only=True)
        old = _head_tool_json(head_tree, {"total": 10.0})
        assert json.dumps(_minus_trace(new)) == json.dumps(_minus_trace(old))

    def test_envelope_bad_inner_key_over_mcp(self):
        r = tools.validate_tool(SRC, json.loads(json.dumps(ENV_BAD)), heuristics_only=True)
        assert r["status"] == "error"
        assert _unknown_list(r["error_message"]) == "vendr_name"

    @pytest.mark.parametrize("data", [
        {1: "x"},
        {"total": 10.0, 2: 3},
        {("t", "u"): 1, "total": 10.0},
    ], ids=["int-key", "int-beside-known", "tuple-key"])
    def test_non_str_keys_never_raise(self, data):
        # README: "The tools never raise". A Python library caller can pass any
        # dict; HEAD returned an envelope for these, so the new check must too.
        r = tools.validate_tool(SRC, data, heuristics_only=True)
        assert isinstance(r, dict)
        assert r["status"] in {"ok", "needs_review", "error"}


# ---------------------------------------------------------------------------
# EB8: Generator path untouched
# ---------------------------------------------------------------------------

class TestEB8GeneratorPathUntouched:
    def _pipe(self):
        gen = Generator(ScriptedLLMClient([json.dumps({"total": 10.0, "note": "x"})]))
        crit = Critic(ScriptedLLMClient([critique_all(0.95, 0.95)]))
        return ReflectivePipeline(gen, crit)

    def test_extra_key_in_generator_reply_parses(self):
        r = self._pipe().extract(SRC)
        assert r.status != "error", getattr(r, "error_message", None)
        assert r.data.total == 10.0
        assert "note" not in r.data.model_dump()

    def test_extract_tool_with_extra_key_reply(self, monkeypatch):
        import reflect_mcp.pipeline as pipeline_module

        def factory(gate_config=None, **kwargs):
            gen = Generator(ScriptedLLMClient([json.dumps({"total": 10.0, "note": "x"})]))
            crit = Critic(ScriptedLLMClient([critique_all(0.95, 0.95)]))
            return ReflectivePipeline(gen, crit, gate_config=gate_config)

        # Another suite module re-imports reflect_mcp submodules, so patch every
        # pipeline-module object the tools module could resolve the factory from.
        targets = {id(pipeline_module): pipeline_module}
        live = sys.modules.get("reflect_mcp.pipeline")
        if live is not None:
            targets[id(live)] = live
        for val in vars(tools).values():
            if isinstance(val, types.ModuleType) and val.__name__ == "reflect_mcp.pipeline":
                targets[id(val)] = val
        for mod in targets.values():
            monkeypatch.setattr(mod, "build_default_pipeline", factory)
        if "build_default_pipeline" in vars(tools):
            monkeypatch.setattr(tools, "build_default_pipeline", factory)
        r = tools.extract_tool(SRC)
        assert r["status"] != "error", r
        assert r["data"]["total"] == 10.0
        assert "note" not in r["data"]

    def test_context_free_model_validate_ignores_extra(self):
        m = InvoiceData.model_validate({"total": 10.0, "x": 1})
        assert m.total == 10.0
        assert "x" not in m.model_dump()


# ---------------------------------------------------------------------------
# EB9: eval and pins
# ---------------------------------------------------------------------------

def _eval(tree: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "eval/run_eval.py"], capture_output=True, text=True,
                          env=_env(tree), cwd=str(tree), timeout=300)


class TestEB9EvalAndPins:
    def test_eval_byte_identical_to_head(self, head_tree):
        new, old = _eval(REPO), _eval(head_tree)
        assert new.returncode == old.returncode
        assert new.stdout == old.stdout
        assert new.stdout.strip()

    def test_cli_line_count_kept(self):
        lines = (REPO / "src" / "reflect_mcp" / "cli.py").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 306  # iter 81: 375 -> 306, duplicate parser builders deleted


# ---------------------------------------------------------------------------
# Docs (acceptance criteria): README §MCP tools and §Batch mode
# ---------------------------------------------------------------------------

class TestDocs:
    def _para(self, startswith: str) -> str:
        text = (REPO / "README.md").read_text(encoding="utf-8")
        paras = [p for p in text.split("\n\n") if p.lstrip().startswith(startswith)]
        assert len(paras) == 1, startswith
        return paras[0]

    @pytest.mark.parametrize("lead", ["The tools never raise", "**Batch mode**"])
    def test_paragraph_documents_unknown_key_rule(self, lead):
        para = self._para(lead)
        assert "not an `InvoiceData` field" in para
        assert "schema error naming the key" in para
        assert "exempt" in para

    def test_batch_paragraph_names_all_three_surfaces(self):
        para = self._para("**Batch mode**")
        assert "exit 2 for `--data`" in para
        assert 'status:"error"` over MCP' in para
