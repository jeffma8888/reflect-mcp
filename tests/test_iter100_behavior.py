"""Black-box behavioral tests for iter-100: ``import reflect_mcp`` is lazy.

The package re-exports its public names through a PEP 562 module
``__getattr__`` so the on-ramp verbs (``--version``, ``--help``,
``--mcp-config``) never import pydantic or the pipeline modules.

  1  bare ``import reflect_mcp`` loads neither pydantic nor the pipeline
  2  ``import reflect_mcp.__main__`` loads none of the heavy modules
  3  ``--version`` bytes unchanged, rc 0; runpy of ``__main__`` keeps pydantic out
  4  ``--help`` / ``--mcp-config`` stdout equals the ac81759 output, same rc,
     and neither verb loads pydantic
  5  ``__all__`` exact set and order; each name is its origin module's object;
     star import binds all 16
  6  the README library import works and does load pydantic (lazy, not removed)
  7  an unknown attribute raises AttributeError naming it; hasattr is False
  8  ``dir()`` lists every public name before any access
  9  the re-exports stay visible to static analysers (TYPE_CHECKING imports)
 10  ``validate --heuristics-only`` (single and batch) and the frozen eval are
     byte-identical to ac81759 (trace ids masked)

Every ``sys.modules`` check runs in a fresh ``sys.executable`` subprocess,
because this suite has long since imported pydantic in-process. The ac81759
fingerprints below were measured once, outside the suite, from an exported
copy of that commit; nothing here reads git at test time. No network, no
keys, no wall-clock bound, and no optional extra is imported.

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests._golden import FROZEN_EVAL_SHA256, frozen_eval, frozen_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_PY = REPO_ROOT / "src" / "reflect_mcp" / "__init__.py"

HEAD_ALL = [
    "InvoiceData",
    "CritiqueResult",
    "FieldVerdict",
    "FlaggedField",
    "ProvenanceSpan",
    "ExtractionResult",
    "EXTRACTION_FIELDS",
    "ReflectivePipeline",
    "build_default_pipeline",
    "Generator",
    "GeneratorError",
    "Critic",
    "CriticError",
    "GateConfig",
    "decide",
    "__version__",
]
LAZY_NAMES = [n for n in HEAD_ALL if n != "__version__"]
# Origin module of each re-export (measured on ac81759 via __module__ / identity).
ORIGIN = {
    "InvoiceData": "schema",
    "CritiqueResult": "schema",
    "FieldVerdict": "schema",
    "FlaggedField": "schema",
    "ProvenanceSpan": "schema",
    "ExtractionResult": "schema",
    "EXTRACTION_FIELDS": "schema",
    "ReflectivePipeline": "pipeline",
    "build_default_pipeline": "pipeline",
    "Generator": "generator",
    "GeneratorError": "generator",
    "Critic": "critic",
    "CriticError": "critic",
    "GateConfig": "gate",
    "decide": "gate",
}
HEAVY = [
    "pydantic",
    "reflect_mcp.schema",
    "reflect_mcp.pipeline",
    "reflect_mcp.generator",
    "reflect_mcp.critic",
    "reflect_mcp.gate",
]

# ac81759 fingerprints (COLUMNS=80, hermetic env, interpreter path masked).
HEAD_HELP_SHA256 = "6379969155d9616fd0ef04d5508fadbba9d0d6a92fd81e1cbca49a17be23927f"
HEAD_MCP_CONFIG = (
    "{\n"
    '  "mcpServers": {\n'
    '    "reflect-mcp": {\n'
    '      "command": "<PY>",\n'
    '      "args": [\n'
    '        "-m",\n'
    '        "reflect_mcp.server"\n'
    "      ],\n"
    '      "env": {\n'
    '        "REFLECT_PROVIDER": "anthropic",\n'
    '        "ANTHROPIC_API_KEY": "<your-api-key>"\n'
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
)
# validate --heuristics-only on frozen row inv-002 (single) and inv-001..008 (batch).
HEAD_VALIDATE_SINGLE_SHA256 = "171b42dbd917054546e037dc706795220fa0ed985eeb51e79c970d70d30b2162"
HEAD_VALIDATE_BATCH_SHA256 = "aba9e02b6bf501e19dc905c06160656accd7535ad7b6e1a686c0abd042dea0b8"


def _env(**extra: str) -> dict[str, str]:
    """Inherited env minus REFLECT_*, *_API_KEY and AWS_*; fixed width; then extras."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("REFLECT_") or k.endswith("_API_KEY") or k.startswith("AWS_"))
    }
    env["COLUMNS"] = "80"
    env.update(extra)
    return env


def _py(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=_env(),
        timeout=120,
    )


def _cli(*args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=_env(),
        timeout=120,
    )


def _runpy_modules(*argv: str) -> tuple[int, str, list[str]]:
    """Run ``reflect_mcp.__main__`` via runpy in a fresh interpreter.

    Returns (exit code, captured stdout of the verb, heavy modules loaded after).
    The census is written to stderr as a JSON line so it never mixes with stdout.
    """
    code = (
        "import io, json, runpy, sys, contextlib\n"
        f"sys.argv = ['reflect-mcp', *{list(argv)!r}]\n"
        "buf = io.StringIO(); rc = 0\n"
        "try:\n"
        "    with contextlib.redirect_stdout(buf):\n"
        "        runpy.run_module('reflect_mcp', run_name='__main__', alter_sys=True)\n"
        "except SystemExit as exc:\n"
        "    rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "sys.stderr.write('CENSUS ' + json.dumps({'rc': rc, 'out': buf.getvalue(), 'heavy': heavy}) + '\\n')\n"
    )
    proc = _py(code)
    assert proc.returncode == 0, proc.stderr
    line = [ln for ln in proc.stderr.splitlines() if ln.startswith("CENSUS ")]
    assert len(line) == 1, proc.stderr
    payload = json.loads(line[0][len("CENSUS ") :])
    return payload["rc"], payload["out"], payload["heavy"]


def _mask_trace(text: str) -> str:
    return re.sub(r"trace-[0-9A-Za-z_-]+", "trace-X", text)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no version line"
    return match.group(1)


# ---------------------------------------------------------------- EB1 / EB2


def test_eb1_bare_import_loads_neither_pydantic_nor_pipeline() -> None:
    proc = _py(
        "import reflect_mcp, sys; "
        "print('pydantic' in sys.modules, 'reflect_mcp.pipeline' in sys.modules)"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "False False\n"


def test_eb1_bare_import_leaves_every_heavy_module_unloaded() -> None:
    proc = _py(f"import json, sys, reflect_mcp; print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


def test_eb2_import_main_loads_no_heavy_module() -> None:
    proc = _py(
        "import json, sys, reflect_mcp.__main__; "
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))"
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


def test_eb2_oracle_discriminates_after_a_public_access() -> None:
    """The census is not vacuous: touching a public name does load the heavy set."""
    proc = _py(
        "import json, sys, reflect_mcp.__main__, reflect_mcp; reflect_mcp.ReflectivePipeline; "
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))"
    )
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads(proc.stdout)
    assert "pydantic" in loaded and "reflect_mcp.pipeline" in loaded


# ---------------------------------------------------------------- EB3


def test_eb3_version_bytes_unchanged_and_exit_zero() -> None:
    proc = _cli("--version")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"reflect-mcp {_pyproject_version()}\n"
    assert proc.stdout == "reflect-mcp 0.1.0\n"  # the ac81759 bytes
    assert proc.stderr == ""


def test_eb3_version_via_runpy_keeps_pydantic_out() -> None:
    rc, out, heavy = _runpy_modules("--version")
    assert rc == 0
    assert out == "reflect-mcp 0.1.0\n"
    assert "pydantic" not in heavy
    assert heavy == []


# ---------------------------------------------------------------- EB4


def _norm_help(text: str) -> str:
    # Python 3.10 argparse titles the section "optional arguments:".
    return text.replace("optional arguments:", "options:")


def test_eb4_help_stdout_matches_ac81759() -> None:
    proc = _cli("--help")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage: reflect-mcp ")
    assert "--mcp-config" in proc.stdout
    assert _sha(_norm_help(proc.stdout)) == HEAD_HELP_SHA256, proc.stdout


def test_eb4_mcp_config_stdout_matches_ac81759() -> None:
    proc = _cli("--mcp-config")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["mcpServers"]["reflect-mcp"]["command"] == sys.executable
    assert proc.stdout.replace(json.dumps(sys.executable)[1:-1], "<PY>") == HEAD_MCP_CONFIG


@pytest.mark.parametrize("flag", ["--help", "--mcp-config"])
def test_eb4_on_ramp_verbs_keep_pydantic_out(flag: str) -> None:
    rc, out, heavy = _runpy_modules(flag)
    assert rc == 0
    assert out.strip(), f"{flag} printed nothing under runpy"
    assert "pydantic" not in heavy
    assert heavy == []


def test_eb4_runpy_output_equals_cli_output() -> None:
    """The runpy census drives the same verb the CLI runs (not a no-op path)."""
    _, out, _ = _runpy_modules("--help")
    assert _sha(_norm_help(out)) == HEAD_HELP_SHA256


# ---------------------------------------------------------------- EB5


def test_eb5_all_is_exact_list_in_order() -> None:
    proc = _py("import json, reflect_mcp; print(json.dumps(list(reflect_mcp.__all__)))")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == HEAD_ALL


def test_eb5_each_name_is_its_origin_modules_object() -> None:
    code = (
        "import importlib, json, reflect_mcp\n"
        f"origin = {ORIGIN!r}\n"
        "bad = [n for n, m in origin.items()\n"
        "       if getattr(reflect_mcp, n) is not getattr(importlib.import_module('reflect_mcp.' + m), n)]\n"
        "print(json.dumps(bad))\n"
    )
    proc = _py(code)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


def test_eb5_repeated_access_returns_the_same_object() -> None:
    proc = _py(
        "import reflect_mcp\n"
        "print(all(getattr(reflect_mcp, n) is getattr(reflect_mcp, n) for n in reflect_mcp.__all__))"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "True\n"


def test_eb5_version_is_a_plain_string_matching_pyproject() -> None:
    proc = _py("import reflect_mcp, sys; v = reflect_mcp.__version__; print(type(v).__name__, v, 'pydantic' in sys.modules)")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"str {_pyproject_version()} False\n"


def test_eb5_star_import_binds_all_sixteen() -> None:
    code = (
        "import json\n"
        "ns = {}\n"
        "exec('from reflect_mcp import *', ns)\n"
        f"print(json.dumps([n for n in {HEAD_ALL!r} if n not in ns]))\n"
    )
    proc = _py(code)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


# ---------------------------------------------------------------- EB6


def test_eb6_readme_library_import_works_and_loads_pydantic() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    snippet = "from reflect_mcp import build_default_pipeline, GateConfig, ExtractionResult"
    assert snippet in readme
    proc = _py(
        f"import sys; {snippet}; "
        "import reflect_mcp.pipeline as p, reflect_mcp.gate as g, reflect_mcp.schema as s; "
        "print('pydantic' in sys.modules, build_default_pipeline is p.build_default_pipeline, "
        "GateConfig is g.GateConfig, ExtractionResult is s.ExtractionResult)"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "True True True True\n"


def test_eb6_lazy_class_is_usable() -> None:
    proc = _py("import reflect_mcp; print(reflect_mcp.GateConfig().__class__.__name__)")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "GateConfig\n"


# ---------------------------------------------------------------- EB7


def test_eb7_unknown_attribute_raises_attribute_error_naming_it() -> None:
    code = (
        "import reflect_mcp\n"
        "try:\n"
        "    reflect_mcp.does_not_exist\n"
        "except AttributeError as exc:\n"
        "    print('AE', 'does_not_exist' in str(exc))\n"
        "else:\n"
        "    print('NO-ERROR')\n"
        "print(hasattr(reflect_mcp, 'does_not_exist'))\n"
    )
    proc = _py(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "AE True\nFalse\n"


def test_eb7_from_import_of_unknown_name_is_import_error() -> None:
    proc = _py(
        "try:\n"
        "    from reflect_mcp import does_not_exist\n"
        "except ImportError as exc:\n"
        "    print('IE', 'does_not_exist' in str(exc))\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "IE True\n"


# ---------------------------------------------------------------- EB8


def test_eb8_dir_lists_every_public_name_before_access() -> None:
    code = (
        "import json, sys, reflect_mcp\n"
        "names = dir(reflect_mcp)\n"
        f"print(json.dumps([n for n in {HEAD_ALL!r} if n not in names]), 'pydantic' in sys.modules)\n"
    )
    proc = _py(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "[] False\n"


# ---------------------------------------------------------------- EB9


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def test_eb9_type_checking_block_imports_every_lazy_name() -> None:
    tree = ast.parse(INIT_PY.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for sub in node.body:
                for inner in ast.walk(sub):
                    if isinstance(inner, ast.ImportFrom):
                        imported.update(a.asname or a.name for a in inner.names)
    missing = [n for n in LAZY_NAMES if n not in imported]
    assert missing == [], f"not imported under TYPE_CHECKING: {missing}"


def test_eb9_version_is_a_module_level_constant() -> None:
    tree = ast.parse(INIT_PY.read_text(encoding="utf-8"))
    found = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(t, ast.Name) and t.id == "__version__"
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert len(found) == 1


# ---------------------------------------------------------------- EB10


def _frozen() -> list[dict]:
    rows = frozen_rows()
    assert [r["id"] for r in rows][:2] == ["inv-001", "inv-002"]
    return rows


def test_eb10_validate_heuristics_only_single_matches_ac81759() -> None:
    row = [r for r in _frozen() if r["id"] == "inv-002"][0]
    proc = _cli("validate", "--heuristics-only", "--data", json.dumps(row["expected"]), stdin=row["text"])
    assert proc.returncode == 0, proc.stderr
    assert "trace-" in proc.stdout  # the mask has something to mask
    assert _sha(_mask_trace(proc.stdout)) == HEAD_VALIDATE_SINGLE_SHA256, proc.stdout


def test_eb10_validate_heuristics_only_batch_matches_ac81759() -> None:
    batch = "".join(
        json.dumps({"id": r["id"], "text": r["text"], "data": r["expected"]}) + "\n" for r in _frozen()
    )
    proc = _cli("validate", "--batch", "--heuristics-only", stdin=batch)
    assert proc.returncode == 0, proc.stderr
    assert len(proc.stdout.splitlines()) == 8
    assert _sha(_mask_trace(proc.stdout)) == HEAD_VALIDATE_BATCH_SHA256, proc.stdout


def test_eb10_frozen_eval_is_byte_identical_and_exits_zero() -> None:
    proc = frozen_eval()
    assert proc.returncode == 0, proc.stderr
    assert _sha(proc.stdout) == FROZEN_EVAL_SHA256


# ---------------------------------------------------------------- hygiene


def test_module_imports_no_optional_extra_and_no_machine_path() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    assert not any(m == "mcp" or m.startswith("mcp.") or m == "reflect_mcp" + ".server" for m in mods)
    assert ("/" + "Users/") not in text and ("/" + "home/") not in text
