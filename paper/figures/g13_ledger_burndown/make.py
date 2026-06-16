"""G13 backend-port burndown: tests still xfail-ledgered vs time.

The migration progress metric, one panel per backend (very different scales).
The all-backend suite stays green during the port via a checked-in
xfail(strict=False) ledger; the number of LEDGERED tests (still failing under a
backend) is the remaining work, and it falls as each subsystem -- potentials,
orbit, action-angle, df -- is ported and its tests move xfail->pass at a
milestone review. The series starts at the FULL suite size (the project-start
anchor at #892's opening: before the backend infra existed, nothing ran under
jax/torch) and burns down toward 0. Measured points are reconstructed exactly
from the git history of tests/backend_xfail.txt + backend_slow_skip.txt on
feat/backends (measure/burndown_from_git.py); re-run that + `run` at each
milestone to extend the series.
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


def make_panel(ax, dates, vals, color, label, total, milestones, show_xticks):
    # anchor (open marker) -> measured points (filled)
    ax.plot(dates, vals, "-", color=color, lw=1.4, zorder=3)
    ax.plot(dates[1:], vals[1:], "o", color=color, ms=4, zorder=4)
    ax.plot(dates[0], vals[0], "o", mfc="white", mec=color, ms=6, zorder=4)
    for m in milestones:
        ax.axvline(_d(m["date"]), color="0.78", lw=0.8, ls="--", zorder=0)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_ylim(0, total * 1.08)
    ax.set_ylabel(f"{label}\ntests still ledgered")
    # current value + start value callouts
    ax.annotate(f"{vals[-1]}", (dates[-1], vals[-1]), textcoords="offset points",
                xytext=(5, 2), fontsize=8, color=color, fontweight="bold")
    ax.annotate(f"{vals[0]}", (dates[0], vals[0]), textcoords="offset points",
                xytext=(4, 4), fontsize=7.5, color="0.4")
    ax.margins(x=0.04)


@figure(
    name="g13_ledger_burndown",
    track="G",
    seed=0,
    description="backend-port burndown per backend: tests still xfail-ledgered vs time",
)
def make():
    rows = _load("burndown.json")
    anchor = _load("anchor.json")
    milestones = _load("milestones.json")
    total = anchor["total"]
    a = _d(anchor["date"])
    dates = [a] + [_d(r["date"]) for r in rows]
    jax = [total] + [r["jax_total"] for r in rows]
    torch = [total] + [r["torch_total"] for r in rows]

    fig, (axt, axj) = plt.subplots(
        2, 1, figsize=(6.2, 4.8), sharex=True, gridspec_kw={"hspace": 0.12}
    )
    make_panel(axt, dates, torch, "#d1495b", "torch", total, milestones, False)
    make_panel(axj, dates, jax, "#1f77b4", "jax", total, milestones, True)

    # milestone labels once, tucked inside the top of the upper panel
    for m in milestones + [{"date": anchor["date"], "label": anchor["label"]}]:
        axt.text(_d(m["date"]), total * 1.04, " " + m["label"], rotation=90,
                 va="top", ha="left", fontsize=5.6, color="0.45", zorder=5)

    fig.suptitle(
        "Backend-port burndown — tests still failing per backend "
        "(target 0 = full suite green at parity)", fontsize=10, y=0.98,
    )
    axj.set_xlabel("date")
    loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
    axj.xaxis.set_major_locator(loc)
    axj.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


if __name__ == "__main__":
    make()
