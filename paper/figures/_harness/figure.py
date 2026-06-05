"""Reproducible figure harness core: the @figure decorator, registry, and the
run / verify / list operations.

Every paper figure is a ``make()`` function decorated with ``@figure(...)``,
living in ``paper/figures/<name>/make.py`` and returning a ``matplotlib.Figure``.
The harness seeds RNGs deterministically, applies the shared style, writes a
byte-stable PDF (CreationDate suppressed), and records sha256 + git sha + env
into ``manifest.json``.

CLI (run from the galpy-jax repo root):

    python -m paper.figures._harness list
    python -m paper.figures._harness run    [name ...]
    python -m paper.figures._harness verify [name ...]

``verify`` re-renders each figure to a temp file and asserts its sha256 matches
the manifest -- the reproducibility gate to run against the *final* galpy code.
Byte-stability holds for a fixed matplotlib version; on a version bump, re-run
``run`` to refresh the manifest (the env stamp records which version produced it).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .style import apply_style  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = _HARNESS_DIR.parent  # paper/figures/
REPO_DIR = FIGURES_DIR.parent.parent  # galpy-jax/
MANIFEST = _HARNESS_DIR / "manifest.json"

_REGISTRY: dict[str, dict] = {}


def figure(name, track=None, seed=0, description=""):
    """Register a figure ``make()`` function under ``name``."""

    def deco(fn):
        if name in _REGISTRY and _REGISTRY[name]["fn"] is not fn:
            raise ValueError(f"duplicate figure name: {name!r}")
        _REGISTRY[name] = dict(
            fn=fn, track=track, seed=seed, description=description,
            module=getattr(fn, "__module__", "?"),
        )
        return fn

    return deco


def _seed_all(seed):
    import random

    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except Exception:
        pass


def discover():
    """Import every ``paper/figures/<name>/make.py`` to populate the registry."""
    for make in sorted(FIGURES_DIR.glob("*/make.py")):
        spec = importlib.util.spec_from_file_location(
            f"_figmake_{make.parent.name}", make
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return _REGISTRY


def _git_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(REPO_DIR),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _env_stamp():
    stamp = {"matplotlib": matplotlib.__version__, "python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "jax", "torch", "galpy"):
        try:
            stamp[mod] = __import__(mod).__version__
        except Exception:
            pass
    return stamp


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _render_to(name, path):
    spec = _REGISTRY[name]
    _seed_all(spec["seed"])
    apply_style()
    fig = spec["fn"]()
    # metadata CreationDate=None suppresses the timestamp -> byte-stable PDF.
    fig.savefig(
        str(path),
        metadata={
            "CreationDate": None,
            "Creator": "galpy-jax figure harness",
            "Producer": "matplotlib",
        },
    )
    plt.close(fig)


def _load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def _targets(names):
    discover()
    names = names or sorted(_REGISTRY)
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise SystemExit(
            f"unknown figure(s): {unknown}; known: {sorted(_REGISTRY)}"
        )
    return names


def cmd_list(_args):
    discover()
    print(f"{'figure':28s} {'track':6s} description")
    for name in sorted(_REGISTRY):
        s = _REGISTRY[name]
        print(f"{name:28s} {str(s['track'] or '-'):6s} {s['description']}")


def cmd_run(args):
    names = _targets(args.names)
    manifest = _load_manifest()
    for name in names:
        outdir = FIGURES_DIR / name
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{name}.pdf"
        _render_to(name, out)
        digest = _sha256(out.read_bytes())
        manifest[name] = {
            "sha256": digest,
            "git": _git_sha(),
            "env": _env_stamp(),
            "track": _REGISTRY[name]["track"],
            "description": _REGISTRY[name]["description"],
        }
        print(f"[run]    {name:28s} -> {out.relative_to(REPO_DIR)}  sha256={digest[:12]}")
    _save_manifest(manifest)


def cmd_verify(args):
    names = _targets(args.names)
    manifest = _load_manifest()
    rc = 0
    for name in names:
        if name not in manifest:
            print(f"[verify] {name}: NOT in manifest (run it first)")
            rc = 1
            continue
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            _render_to(name, tmp.name)
            digest = _sha256(Path(tmp.name).read_bytes())
        exp = manifest[name]["sha256"]
        if digest == exp:
            print(f"[verify] {name}: OK ({digest[:12]})")
        else:
            mpl_then = manifest[name].get("env", {}).get("matplotlib", "?")
            print(
                f"[verify] {name}: MISMATCH expected {exp[:12]} got {digest[:12]} "
                f"(manifest matplotlib={mpl_then}, now={matplotlib.__version__})"
            )
            rc = 1
    raise SystemExit(rc)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m paper.figures._harness",
        description="galpy-jax reproducible figure harness",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="(re)generate figures + update manifest")
    pr.add_argument("names", nargs="*")
    pr.set_defaults(func=cmd_run)
    pv = sub.add_parser("verify", help="re-render and assert sha256 matches manifest")
    pv.add_argument("names", nargs="*")
    pv.set_defaults(func=cmd_verify)
    pl = sub.add_parser("list", help="list registered figures")
    pl.set_defaults(func=cmd_list)
    args = p.parse_args(argv)
    args.func(args)
