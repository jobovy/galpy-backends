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

2026-10-06: dashed lines = the --jit rows (jax-jit / torch-jit ledger entries, from
the mode's introduction on 2026-07-26); y is symlog (the first weeks fall by
thousands, the tail is single digits); dates are AUTHOR dates (feat/backends is
rebased onto main periodically, which rewrites committer dates).
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


def make_panel(ax, dates, vals, color, label, total, milestones, show_xticks,
               jit=None):
    # anchor (open marker) -> measured points (filled)
    ax.plot(dates, vals, "-", color=color, lw=1.4, zorder=3, label="eager")
    ax.plot(dates[1:], vals[1:], "o", color=color, ms=2.5, zorder=4)
    ax.plot(dates[0], vals[0], "o", mfc="white", mec=color, ms=6, zorder=4)
    if jit is not None:  # the --jit rows, from the day the mode was added
        jd, jv = jit
        ax.plot(jd, jv, "--", color=color, lw=1.0, alpha=0.75, zorder=3,
                label="under jit")
        ax.annotate(f"{jv[-1]}", (jd[-1], jv[-1]), textcoords="offset points",
                    xytext=(5, -9), fontsize=7, color=color, alpha=0.8)
        ax.legend(loc="lower left", fontsize=7, frameon=False)
    for m in milestones:
        ax.axvline(_d(m["date"]), color="0.78", lw=0.8, ls="--", zorder=0)
    # symlog: the first weeks fall by ~1000s, the tail is single digits
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(0, total * 1.6)
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
    # jit rows exist from the --jit mode's introduction on
    j0 = next(i for i, r in enumerate(rows)
              if r.get("jax_jit_total", 0) + r.get("torch_jit_total", 0) > 0)
    jdates = [_d(r["date"]) for r in rows[j0:]]
    jjax = (jdates, [r["jax_jit_total"] for r in rows[j0:]])
    jtorch = (jdates, [r["torch_jit_total"] for r in rows[j0:]])

    fig, (axt, axj) = plt.subplots(
        2, 1, figsize=(6.2, 5.6), sharex=True, gridspec_kw={"hspace": 0.12}
    )
    make_panel(axt, dates, torch, "#d1495b", "torch", total, milestones, False,
               jit=jtorch)
    make_panel(axj, dates, jax, "#1f77b4", "jax", total, milestones, True,
               jit=jjax)

    # milestones: numbered at the top of the upper panel, keyed below the axes
    ms = [{"date": anchor["date"], "label": anchor["label"]}] + milestones
    for i, m in enumerate(ms, 1):
        axt.annotate(str(i), (_d(m["date"]), 1.0), xycoords=("data", "axes fraction"),
                     xytext=(0, 2), textcoords="offset points", ha="center",
                     va="bottom", fontsize=6.5, color="0.35")
    key = [f"{i}: {m['date']}  {m['label']}" for i, m in enumerate(ms, 1)]
    half = (len(key) + 1) // 2
    for col, chunk in enumerate((key[:half], key[half:])):
        fig.text(0.02 + 0.5 * col, 0.005, "\n".join(chunk), fontsize=6,
                 color="0.3", va="bottom", ha="left")

    fig.suptitle(
        "Backend-port burndown — tests still failing per backend "
        "(target 0 = full suite green at parity)", fontsize=10, y=0.98,
    )
    axj.set_xlabel("date")
    loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
    axj.xaxis.set_major_locator(loc)
    axj.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.25)
    return fig


if __name__ == "__main__":
    make()
