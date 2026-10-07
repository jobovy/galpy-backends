"""Run every code example of the paper (one file per snippet in the .tex).

Each example is run in a fresh interpreter, because the examples set global
state (jax's x64 flag, backend contexts) and some patch galpy to count calls.
An example passes when it exits 0; the checks of what the paper claims are
the assertions inside each file. Examples that put arrays on a GPU
(``device='cuda'``) are skipped unless CUDA is available.
"""

import glob
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = sorted(
    os.path.basename(f)
    for f in glob.glob(os.path.join(HERE, "*_[0-9][0-9]_*.py"))
)
BEGIN, END = "# --- paper snippet begins ---", "# --- paper snippet ends ---"
MAXLEN = 56  # characters per line in a 3.4in column (minted, footnotesize)


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        return False


def _snippet(name):
    with open(os.path.join(HERE, name)) as f:
        src = f.read()
    m = re.search(re.escape(BEGIN) + r"\n(.*?)\n[ ]*" + re.escape(END), src, re.S)
    assert m, f"{name} has no paper-snippet markers"
    lines = m.group(1).split("\n")
    indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
    return src, [l[indent:] for l in lines]


@pytest.mark.parametrize("name", EXAMPLES)
def test_snippet_fits_the_column(name):
    _, lines = _snippet(name)
    long = [l for l in lines if len(l) > MAXLEN]
    assert not long, f"lines longer than {MAXLEN} characters: {long}"


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs(name):
    src, _ = _snippet(name)
    if "device='cuda'" in src and not _cuda_available():
        pytest.skip("needs a CUDA GPU")
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, name)],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f"{name} failed:\n--- stdout ---\n{proc.stdout[-4000:]}"
        f"\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
    assert proc.stdout.rstrip().endswith("OK"), proc.stdout[-2000:]
