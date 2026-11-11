"""Black-box behavioral tests for iter-80: a CI workflow that runs the offline
suite and ``make eval`` on Python 3.10 (the ``requires-python`` floor) and
3.13, plus the isolation fix that stops ``TestOnlyServerImportsMcp`` leaking
orphaned package attributes (``reflect_mcp.<n>`` != ``sys.modules[...]``).

Covers the 10 Expected Behaviors from pm.md:
  1. ``.github/workflows/ci.yml`` exists; ``on:`` has push(main),
     pull_request, workflow_dispatch.
  2. top-level ``permissions: contents: read``; ``timeout-minutes`` in
     [5, 30]; ``fail-fast: false``.
  3. matrix ``python-version`` is exactly ["3.10", "3.13"]; first entry is the
     pyproject floor (read by regex -- no tomllib, so this runs on 3.10).
  4. steps in order: checkout@v4, setup-uv@v6 (python-version from matrix),
     ``uv sync --locked --extra server``, ``uv run pytest``, ``make eval``.
  5. every ``run:`` is ``make <existing target>`` or ``uv ...``; ``make eval``
     resolves to ``eval/run_eval.py``; no secrets / provider env vars.
  6. no machine paths, no owner/org name, ``runs-on: ubuntu-latest``.
  7. after TestOnlyServerImportsMcp, every loaded submodule attribute on the
     package *is* its ``sys.modules`` entry (and the probe discriminates
     against the iter-79 test file).
  8. ``mock.patch("reflect_mcp.pipeline.build_default_pipeline")`` lands on
     the ``sys.modules`` module object after the sweep.
  9. no ``src/``/``eval/`` diff vs iter-79; eval stdout identical to HEAD.
 10. README §9 CI line; §11 no longer lists offline CI as future work.

Isolation contract: the workflow is pinned by text/regex only (PyYAML is not a
dependency); the product is exercised through its public package / eval
script. No implementation source was read.

``BASE_SHA`` is the iter-79 commit; comparisons against it skip when it is not
reachable (e.g. on an archive copy or a shallow clone).
"""

from __future__ import annotations

import importlib.util
import os
import pkgutil
import re
import shutil
import subprocess
import sys
import tarfile
import io
from pathlib import Path
from unittest import mock

import pytest

import reflect_mcp

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
BASE_SHA = "83ce30c"


def _ci() -> str:
    assert CI.is_file(), f"missing {CI.relative_to(ROOT)}"
    text = CI.read_text(encoding="utf-8")
    assert text.strip(), "ci.yml is empty"
    return text


def _lines() -> list[str]:
    # Drop full-line comments; keep indentation.
    return [ln for ln in _ci().splitlines() if not ln.lstrip().startswith("#")]


def _top_block(key: str) -> list[str]:
    """Lines of a column-0 mapping block ``key:`` (until the next column-0 key)."""
    lines = _lines()
    pat = re.compile(rf"""^["']?{re.escape(key)}["']?\s*:(.*)$""")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            out = [m.group(1)]
            for nxt in lines[i + 1:]:
                if nxt.strip() and not nxt[0].isspace():
                    break
                out.append(nxt)
            return out
    raise AssertionError(f"no top-level `{key}:` in ci.yml")


def _strip_q(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _steps() -> list[tuple[str, str]]:
    """Ordered (kind, value) for every ``uses:``/``run:`` key, run blocks expanded."""
    lines = _lines()
    out: list[tuple[str, str]] = []
    pat = re.compile(r"^(\s*)(?:-\s+)?(uses|run)\s*:\s*(.*?)\s*$")
    i = 0
    while i < len(lines):
        m = pat.match(lines[i])
        if m:
            kind, val = m.group(2), m.group(3)
            if kind == "run" and val[:1] in ("|", ">"):
                ind = len(m.group(1))
                cmds = []
                j = i + 1
                while j < len(lines) and (
                    not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > ind
                ):
                    if lines[j].strip():
                        cmds.append(lines[j].strip())
                    j += 1
                out.append((kind, "\n".join(cmds)))
                i = j
                continue
            out.append((kind, _strip_q(val)))
        i += 1
    return out


def _run_cmds() -> list[str]:
    cmds = []
    for kind, val in _steps():
        if kind == "run":
            cmds.extend(c for c in val.split("\n") if c.strip())
    return cmds


def _make_targets() -> set[str]:
    text = MAKEFILE.read_text(encoding="utf-8")
    targets: set[str] = set()
    for ln in text.splitlines():
        m = re.match(r"^\.PHONY\s*:(.*)$", ln)
        if m:
            targets.update(m.group(1).split())
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", ln)
        if m and not m.group(1).startswith("."):
            targets.add(m.group(1))
    return targets


# ---------------------------------------------------------------------------
# EB1: file + triggers
# ---------------------------------------------------------------------------

class TestEB1Triggers:
    def test_file_exists_nonempty(self):
        assert _ci().strip()

    def _on(self) -> str:
        return "\n".join(_top_block("on"))

    def test_push_main(self):
        on = self._on()
        assert re.search(r"^\s*push\s*:", on, re.M), on
        # branch main listed under push (inline or block list)
        push = on[re.search(r"^\s*push\s*:", on, re.M).start():]
        push = re.split(r"^\s{0,4}(?:pull_request|workflow_dispatch)\s*:", push, flags=re.M)[0]
        assert re.search(r"branches\s*:", push), push
        assert re.search(r"""(\[|-\s*|,\s*)["']?main["']?""", push), push

    def test_pull_request(self):
        assert re.search(r"^\s*pull_request\s*:", self._on(), re.M)

    def test_workflow_dispatch(self):
        assert re.search(r"^\s*workflow_dispatch\s*:", self._on(), re.M)


# ---------------------------------------------------------------------------
# EB2: permissions / timeout / fail-fast
# ---------------------------------------------------------------------------

class TestEB2Hardening:
    def test_top_level_permissions_contents_read(self):
        block = "\n".join(_top_block("permissions"))
        assert re.search(r"contents\s*:\s*read\b", block), block

    def test_timeout_minutes_bounded(self):
        vals = re.findall(r"^\s*timeout-minutes\s*:\s*(\S+)\s*$", _ci(), re.M)
        assert vals, "no timeout-minutes"
        for v in vals:
            assert re.fullmatch(r"\d+", v), v
            assert 5 <= int(v) <= 30, v

    def test_fail_fast_false(self):
        assert re.search(r"^\s*fail-fast\s*:\s*false\s*$", _ci(), re.M)
        assert not re.search(r"^\s*fail-fast\s*:\s*true\b", _ci(), re.M)


# ---------------------------------------------------------------------------
# EB3: matrix == ["3.10", "3.13"], first == pyproject floor
# ---------------------------------------------------------------------------

def _matrix_versions() -> list[str]:
    lines = _lines()
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)python-version\s*:\s*(.*?)\s*$", ln)
        if not m or "matrix." in m.group(2):
            continue
        val = m.group(2)
        if val.startswith("["):
            assert val.endswith("]"), val
            return [_strip_q(x) for x in val[1:-1].split(",") if x.strip()]
        if val == "":
            ind = len(m.group(1))
            out = []
            for nxt in lines[i + 1:]:
                mm = re.match(r"^(\s*)-\s*(.+?)\s*$", nxt)
                if mm and len(mm.group(1)) >= ind:
                    out.append(_strip_q(mm.group(2)))
                    continue
                break
            return out
    raise AssertionError("no matrix python-version list in ci.yml")


class TestEB3Matrix:
    def test_versions_exact(self):
        assert _matrix_versions() == ["3.10", "3.13"]

    def test_versions_quoted(self):
        # unquoted 3.10 would be the YAML float 3.1
        assert re.search(r"""["']3\.10["']""", _ci()), "3.10 must be quoted (YAML would read 3.1)"

    def test_first_entry_is_pyproject_floor(self):
        m = re.search(r'^requires-python\s*=\s*">=\s*([0-9.]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
        assert m, "pyproject has no requires-python >= floor"
        assert m.group(1) == "3.10"
        assert _matrix_versions()[0] == m.group(1)


# ---------------------------------------------------------------------------
# EB4: steps in order
# ---------------------------------------------------------------------------

EXPECTED = [
    ("uses", "actions/checkout@v4"),
    ("uses", "astral-sh/setup-uv@v6"),
    ("run", "uv sync --locked --extra server"),
    ("run", "uv run pytest"),
    ("run", "make eval"),
]


class TestEB4Steps:
    def test_order(self):
        steps = _steps()
        idx = []
        for want in EXPECTED:
            assert want in steps, f"{want} missing from {steps}"
            idx.append(steps.index(want))
        assert idx == sorted(idx), f"wrong order: {steps}"

    def test_run_values_exact(self):
        runs = [v for k, v in _steps() if k == "run"]
        for want in ("uv sync --locked --extra server", "uv run pytest", "make eval"):
            assert want in runs, runs

    def test_setup_uv_python_version_from_matrix(self):
        text = _ci()
        i = text.index("astral-sh/setup-uv@v6")
        nxt = re.search(r"^\s*-\s+(name|uses|run)\s*:", text[i:].split("\n", 1)[1], re.M)
        seg = text[i:].split("\n", 1)[1]
        seg = seg[: nxt.start()] if nxt else seg
        assert re.search(r"python-version\s*:\s*\$\{\{\s*matrix\.python-version\s*\}\}", seg), seg


# ---------------------------------------------------------------------------
# EB5: run commands, make eval mapping, no secrets
# ---------------------------------------------------------------------------

class TestEB5Commands:
    def test_every_run_is_make_target_or_uv(self):
        cmds = _run_cmds()
        assert cmds
        targets = _make_targets()
        for c in cmds:
            m = re.fullmatch(r"make\s+(\S+)", c)
            if m:
                assert m.group(1) in targets, f"{c!r}: no such Makefile target"
            else:
                assert re.match(r"uv\s", c), f"{c!r} is neither make <target> nor uv ..."

    def test_make_eval_runs_eval_script(self):
        text = MAKEFILE.read_text(encoding="utf-8")
        m = re.search(r"^eval\s*:.*\n((?:\t.*\n?)+)", text, re.M)
        assert m, "Makefile has no eval rule"
        assert "eval/run_eval.py" in m.group(1)
        assert "--live" not in m.group(1)

    def test_no_secrets_literal(self):
        assert "secrets." not in _ci()

    @pytest.mark.parametrize("name", ["REFLECT_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_"])
    def test_no_provider_env(self, name):
        assert name not in _ci()

    def test_no_api_key_like_env(self):
        assert not re.search(r"(?i)api[_-]?key|token\s*:", _ci())


# ---------------------------------------------------------------------------
# EB6: publication safety + runner
# ---------------------------------------------------------------------------

class TestEB6PublicationSafe:
    @pytest.mark.parametrize("frag", ["/Users/", "/home/", "C:\\"])
    def test_no_machine_paths(self, frag):
        assert frag not in _ci()

    def test_no_owner_name(self):
        text = _ci()
        assert "github.com/" not in text
        assert "jincm" not in text.lower()
        try:
            url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            url = ""
        m = re.search(r"[:/]([^/:]+)/[^/]+?(?:\.git)?$", url)
        if m and len(m.group(1)) > 2:
            assert m.group(1).lower() not in text.lower()

    def test_runs_on_ubuntu_latest(self):
        vals = re.findall(r"^\s*runs-on\s*:\s*(\S+)\s*$", _ci(), re.M)
        assert vals and all(_strip_q(v) == "ubuntu-latest" for v in vals), vals


# ---------------------------------------------------------------------------
# EB7 / EB8: the sweep no longer orphans package attributes
# ---------------------------------------------------------------------------

def _load_iter73(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_sweep():
    mod = _load_iter73(ROOT / "tests" / "test_iter73_behavior.py", "_iter80_iter73_copy")
    cls = mod.TestOnlyServerImportsMcp
    with pytest.MonkeyPatch.context() as mp:
        cls().test_other_modules_import_without_mcp(mp)


def _orphans() -> list[str]:
    bad = []
    for m in pkgutil.iter_modules(reflect_mcp.__path__):
        full = f"reflect_mcp.{m.name}"
        if full in sys.modules and hasattr(reflect_mcp, m.name):
            if getattr(reflect_mcp, m.name) is not sys.modules[full]:
                bad.append(m.name)
    return bad


PROBE = '''
import pkgutil, sys
import reflect_mcp

def test_probe():
    bad = [m.name for m in pkgutil.iter_modules(reflect_mcp.__path__)
           if f"reflect_mcp.{m.name}" in sys.modules and hasattr(reflect_mcp, m.name)
           and getattr(reflect_mcp, m.name) is not sys.modules[f"reflect_mcp.{m.name}"]]
    assert not bad, "ORPHANS=" + ",".join(sorted(bad))
'''


def _session(tmp_path: Path, iter73_text: str) -> subprocess.CompletedProcess:
    tdir = tmp_path / "tests"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "test_iter73_behavior.py").write_text(iter73_text, encoding="utf-8")
    (tdir / "test_zz_probe.py").write_text(PROBE, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(tdir)])
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=",
         "--rootdir", str(tmp_path), "-c", str(tmp_path / "pytest.ini"),
         f"{tdir / 'test_iter73_behavior.py'}::TestOnlyServerImportsMcp",
         str(tdir / "test_zz_probe.py")],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=180,
    )


class TestEB7Isolation:
    def test_sweep_still_passes_and_lists_tools(self):
        _run_sweep()  # raises if its own assertions (incl. "tools" in names) fail
        text = (ROOT / "tests" / "test_iter73_behavior.py").read_text(encoding="utf-8")
        body = text[text.index("class TestOnlyServerImportsMcp"):]
        body = body[: body.find("\nclass ", 1)] if "\nclass " in body[1:] else body
        assert re.search(r"""assert\s+["']tools["']\s+in\s+names""", body)

    def test_no_orphaned_attributes_after_sweep(self):
        _run_sweep()
        assert _orphans() == []

    def test_package_attr_pipeline_and_tools_identity(self):
        _run_sweep()
        assert reflect_mcp.pipeline is sys.modules["reflect_mcp.pipeline"]
        assert reflect_mcp.tools is sys.modules["reflect_mcp.tools"]

    def test_fresh_session_probe_passes_on_worktree(self, tmp_path):
        r = _session(tmp_path, (ROOT / "tests" / "test_iter73_behavior.py").read_text(encoding="utf-8"))
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        assert "2 passed" in r.stdout, r.stdout[-2000:]

    def test_probe_discriminates_against_base(self, tmp_path):
        try:
            old = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{BASE_SHA}:tests/test_iter73_behavior.py"],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pytest.skip("git unavailable")
        if old.returncode != 0:
            pytest.skip(f"{BASE_SHA} not reachable")
        r = _session(tmp_path, old.stdout)
        assert r.returncode != 0, r.stdout[-2000:]
        assert "1 passed" in r.stdout and "1 failed" in r.stdout, r.stdout[-2000:]
        m = re.search(r"ORPHANS=([A-Za-z_,]+)", r.stdout)
        assert m, r.stdout[-2000:]
        names = m.group(1).split(",")
        assert "pipeline" in names and "tools" in names, names


class TestEB8MockPatchTarget:
    def test_patch_lands_on_sys_modules_object(self):
        _run_sweep()
        sentinel = mock.MagicMock(name="bdp")
        with mock.patch("reflect_mcp.pipeline.build_default_pipeline", sentinel) as p:
            assert sys.modules["reflect_mcp.pipeline"].build_default_pipeline is p
            assert p is sentinel
        assert sys.modules["reflect_mcp.pipeline"].build_default_pipeline is not sentinel

    def test_patch_tools_module_too(self):
        _run_sweep()
        tools = sys.modules["reflect_mcp.tools"]
        names = [n for n in dir(tools) if callable(getattr(tools, n)) and not n.startswith("_")]
        assert names
        n = names[0]
        with mock.patch(f"reflect_mcp.tools.{n}") as p:
            assert getattr(sys.modules["reflect_mcp.tools"], n) is p


# ---------------------------------------------------------------------------
# EB9: no src/eval change; eval stdout identical to base
# ---------------------------------------------------------------------------

def _base_reachable() -> bool:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{BASE_SHA}^{{commit}}"],
                           capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def _eval(src: Path, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in {"REFLECT_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
    env["PYTHONPATH"] = os.pathsep.join([str(src), str(cwd), str(cwd / "eval")])
    return subprocess.run([sys.executable, "eval/run_eval.py"], cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=300)


class TestEB9NoProductChange:
    def test_no_src_eval_diff_vs_base(self):
        if not _base_reachable():
            pytest.skip(f"{BASE_SHA} not reachable")
        # iter 81: compare base to iter 80's own commit, not the working tree, so later
        # iterations' src changes do not trip iter 80's "no product change" claim.
        r = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", BASE_SHA, "2bb26a1", "--", "src", "eval"],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == ""

    def test_eval_passes_with_threshold_line(self):
        r = _eval(ROOT / "src", ROOT)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        # pm.md quotes "PASS: field accuracy 91.4% >= 90.0%"; the real line reads
        # ">= required 90.0%", so match the numbers, not the paraphrase.
        # iter 83: inv-011 moved the live accuracy to 90.9%.
        assert re.search(r"^PASS: field accuracy 90\.9% >= (required )?90\.0%$", r.stdout, re.M), r.stdout[-600:]
        assert re.search(r"^PASS: silent-wrong fields 0\b", r.stdout, re.M), r.stdout[-600:]

    def test_eval_stdout_identical_to_base(self, tmp_path):
        if not _base_reachable():
            pytest.skip(f"{BASE_SHA} not reachable")
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_SHA, "src", "eval"],
                              capture_output=True, timeout=60).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
            tf.extractall(tmp_path)
        head = _eval(tmp_path / "src", tmp_path)
        # iter 83: compare base to iter 80's own commit, not the working tree, so a later
        # golden row (inv-011) does not trip iter 80's "eval unchanged" claim.
        ship = tmp_path / "ship"
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", "2bb26a1", "src", "eval"],
                              capture_output=True, timeout=60).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
            tf.extractall(ship)
        wt = _eval(ship / "src", ship)
        assert head.returncode == wt.returncode == 0
        assert wt.stdout == head.stdout


# ---------------------------------------------------------------------------
# EB10: README
# ---------------------------------------------------------------------------

def _section(title_prefix: str) -> str:
    text = README.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(title_prefix)}.*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert m, f"README has no section {title_prefix}"
    return m.group(1)


class TestEB10Readme:
    def _ci_line(self) -> str:
        sec = _section("9.")
        after = sec.split("make cov", 1)
        assert len(after) == 2, "no make cov in §9"
        rest = after[1].split("```", 1)
        assert len(rest) == 2, "make cov code block not closed"
        para = [ln for ln in rest[1].split("\n") if ln.strip()]
        assert para, "nothing after the code block"
        return para[0]

    def test_line_directly_after_code_block(self):
        line = self._ci_line()
        assert ".github/workflows/ci.yml" in line
        assert "make eval" in line
        assert "3.10" in line and "3.13" in line

    def test_code_block_is_install_test_cov(self):
        sec = _section("9.")
        block = sec.split("```", 2)[1]
        for t in ("make install", "make test", "make cov"):
            assert t in block

    def test_no_url_or_badge(self):
        line = self._ci_line()
        assert "http" not in line and "github.com" not in line
        assert "![" not in line and "badge" not in line.lower()
        assert not re.search(r"github\.com/[^/\s]+/reflect-mcp/actions", README.read_text(encoding="utf-8"))

    def test_section11_no_longer_future_offline_ci(self):
        sec = _section("11.")
        assert "- **Offline eval in CI** on every PR" not in sec
        assert not re.search(r"\*\*Offline eval in CI\*\*", sec)

    def test_section11_keeps_live_eval(self):
        sec = _section("11.")
        assert "periodic **live** eval to catch model drift" in sec
