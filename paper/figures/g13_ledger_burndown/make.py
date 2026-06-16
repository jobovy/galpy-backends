"""G13 backend-port burndown: tests still xfail-ledgered vs time.

The migration progress metric. The all-backend suite stays green during the port
via a checked-in xfail(strict=False) ledger; the number of LEDGERED tests (those
still failing under jax / torch) is the remaining work, and it falls as each
subsystem -- potentials, orbit, action-angle, df -- is ported and its tests move
xfail->pass at a milestone review. Data is reconstructed exactly from the git
history of tests/backend_xfail.txt + backend_slow_skip.txt on feat/backends
(measure/burndown_from_git.py); re-run that at each milestone to extend the
series, then `run` this to refresh the byte-stable PDF.
"""

import datetime
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from paper.figures._harness.figure import figure

_DATA = Path(__file__).resolve().parent / "data"


def _load(name):
    return json.loads((_DATA / name).read_text())


def _d(s):
    return datetime.date.fromisoformat(s)


@figure(
    name="g13_ledger_burndown",
    track="G",
    seed=0,
    description="backend-port burndown: tests still xfail-ledgered (jax/torch) vs time",
)
def make():
    rows = _load("burndown.json")
    milestones = _load("milestones.json")
    dates = [_d(r["date"]) for r in rows]
    jax = [r["jax_total"] for r in rows]
    torch = [r["torch_total"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(dates, torch, "-o", color="#d1495b", label="torch")
    ax.plot(dates, jax, "-o", color="#1f77b4", label="jax")
    # annotate the start value so a sparse early series is still legible
    ax.annotate(
        f"{torch[0]}", (dates[0], torch[0]), textcoords="offset points",
        xytext=(4, 4), fontsize=8, color="#d1495b",
    )
    ax.annotate(
        f"{jax[0]}", (dates[0], jax[0]), textcoords="offset points",
        xytext=(4, -10), fontsize=8, color="#1f77b4",
    )

    ymax = max(torch + jax)
    for m in milestones:
        x = _d(m["date"])
        ax.axvline(x, color="0.7", lw=0.8, ls="--", zorder=0)
        ax.text(
            x, ymax * 0.92, " " + m["label"], rotation=90, va="top", ha="left",
            fontsize=6.0, color="0.45",
        )

    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_ylim(0, ymax * 1.08)
    ax.set_ylabel("tests still xfail-ledgered\n(remaining backend port work)")
    ax.set_xlabel("date")
    ax.set_title(
        "Backend-port burndown (target: 0 = full suite green at parity)", pad=10
    )
    loc = mdates.AutoDateLocator(minticks=2, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.margins(x=0.02)
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    make()
