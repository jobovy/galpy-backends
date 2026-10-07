"""G13 -- backend port of galpy's existing test suite: tests passing under each
backend over time.

Measured by measure/suite_from_git.py: at every commit that touched a ledger
file (backend_xfail.txt, backend_slow_skip.txt, backend_skip.txt) the existing
suite (tests/ minus the test_backend_*.py modules) is collected with pytest at
that commit's ORIGINAL PR head (refs/pull/N/head -- feat/backends is periodically
rebased onto main, and only the rebased tip is guaranteed to import), and each
ledger entry counts as the number of collected tests it matches:

  * a whole-file entry (the jax test_orbit/test_orbits defers of June, the
    streamdf/streamTrack defers of July-August) counts every test in the file,
    not 1 -- the raw entry count understated jax's early ledger by ~700;
  * a parametrized function listed without its parameters counts all of them,
    so splitting one entry into per-parameter entries (2026-06-19, +484 raw
    torch entries) changes nothing.

passing = suite - ledgered (xfail + slow-skip) - permanently skipped, per mode
(eager; --jit dashed, from that mode's introduction). CAVEAT: the --jit series
subtracts only the "<backend>-jit" rows, but under --jit the conftest also applies
the eager xfail and permanent-skip rows (and, before #1279 on 2026-08-09, the eager
slow-skip rows); suite.json does not record that union, so the --jit line is an
upper bound and sits on the suite line where the real one would follow the eager. The suite grows from 2466
(the pre-migration anchor) to 4275 tests: new tests that passed straight away
never entered the ledger, which a ledger count alone cannot show. The bottom
panel is the separate, dedicated backend-test modules (8585 tests in 88 files
at the end). Six rewritten-history commits without a PR head collected nothing
and are dropped.

measure/burndown_from_git.py (raw entry counts) is kept for reference.
"""

import datetime
import json
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from paper.figures._harness import layout
from paper.figures._harness.figure import figure

_DATA = Path(__file__).resolve().parent / "data"
# journal version (default): no suptitle, short callouts, milestone key without
# dates (the axis gives them). GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


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


def _pts():
    """Snapshots with a successful collection (the few rewritten-history commits
    without a PR head to collect at collected nothing; they are dropped)."""
    rows = [r for r in _load("suite.json") if r["existing"] > 0
            # a PR head without the ledger files (#1527, main-based): no entries
            # in any mode, so its counts are not of the ledgered branch
            and sum(r["entries"].values()) > 0]
    # author dates are not monotonic in commit order across rebases: sort (stable)
    return sorted(rows, key=lambda r: r["date"])


@figure(
    name="g13_ledger_burndown",
    track="G",
    seed=0,
    description="backend port of the existing test suite: tests passing per backend (eager, jit) vs time, ledger expanded to the tests it covers; plus the dedicated backend-test modules",
)
def make():
    rows = _pts()
    anchor = _load("anchor.json")
    milestones = _load("milestones.json")
    a = _d(anchor["date"])
    dates = [_d(r["date"]) for r in rows]
    suite = [r["existing"] for r in rows]

    # print size: 7.0 in wide (figure*), text 8-10 pt
    fig, axes = plt.subplots(
        3, 1, figsize=layout.figsize(PAPER, (7.0, 5.6), (6.4, 7.4)), sharex=True,
        gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1, 0.6]},
    )

    def _f(paper_pt, detailed_pt):
        return paper_pt if PAPER else detailed_pt

    # --jit mode exists from the first snapshot with a jit row in either backend
    j0 = next(i for i, r in enumerate(rows)
              if any(r["ledgered"][m] + r["skipped"][m] for m in ("jax-jit", "torch-jit")))
    for ax, mode, color, jcolor, label in (
        (axes[0], "torch", "#d1495b", "#6b1422", "torch"),
        (axes[1], "jax", "#1f77b4", "#0b2f52", "jax"),
    ):
        passing = [r["existing"] - r["ledgered"][mode] - r["skipped"][mode] for r in rows]
        # anchor: before the backend infrastructure nothing ran under jax/torch
        dd, ss, pp = [a] + dates, [anchor["total"]] + suite, [0] + passing
        ax.plot(dd, ss, "-", color="0.45", lw=1.2, label="existing test suite")
        ax.fill_between(dd, pp, ss, color=color, alpha=0.13, lw=0, step=None,
                        label="still ledgered (xfail / slow-skip) or skipped")
        ax.plot(dates, passing, "-", color=color, lw=1.8, label=f"passing under {label}")
        # no data between the pre-migration anchor and the first ledger snapshot
        ax.plot([a, dates[0]], [0, passing[0]], ":", color=color, lw=1.0)
        ax.plot(a, 0, "o", mfc="white", mec=color, ms=6)
        pj = [r["existing"] - r["ledgered"][f"{mode}-jit"] - r["skipped"][f"{mode}-jit"]
              for r in rows[j0:]]
        # on top of the eager and suite lines, which it meets for long stretches:
        # a darker shade, a short dash, a thinner line
        ax.plot(dates[j0:], pj, ls=(0, (2.5, 1.5)), color=jcolor, lw=0.9, zorder=5,
                label=f"passing under {label} --jit")
        last = rows[-1]
        ax.annotate(
            f"{passing[-1]} / {suite[-1]}" + ("" if PAPER else
            f"\n({last['ledgered'][mode]} ledgered, {last['skipped'][mode]} skipped)"),
            (dates[-1], passing[-1]), textcoords="offset points",
            xytext=(-4, -16 if PAPER else -26), ha="right", fontsize=_f(8, 7), color=color,
        )
        for m in milestones:
            ax.axvline(_d(m["date"]), color="0.82", lw=0.7, ls="--", zorder=0)
        ax.set_ylim(0, max(ss) * 1.12)
        ax.set_ylabel(f"{label}\nexisting-suite tests")
        ax.legend(loc="lower right", fontsize=_f(8.5, 6.5), frameon=False)
    bkrows = [(dt, r["backend_modules"]) for dt, r in zip(dates, rows) if r["backend_modules"] > 0]
    bkd, bk = [x for x, _ in bkrows], [y for _, y in bkrows]
    axes[2].plot(bkd, bk, "-", color="#2a9d8f", lw=1.8)
    axes[2].set_ylim(0, max(bk) * 1.15)
    axes[2].set_ylabel("tests in the\nbackend modules")
    axes[2].annotate(f"{bk[-1]}" if PAPER
                     else f"{bk[-1]} tests in {_N_BACKEND_MODULES} test_backend_*.py modules",
                     (bkd[-1], bk[-1]), textcoords="offset points", xytext=(-4, -14),
                     ha="right", fontsize=_f(8, 7), color="#2a9d8f")
    for m in milestones:
        axes[2].axvline(_d(m["date"]), color="0.82", lw=0.7, ls="--", zorder=0)

    # milestones: numbered at the top of the upper panel, keyed below the axes
    ms = [{"date": anchor["date"], "label": anchor["label"]}] + milestones
    for i, m in enumerate(ms, 1):
        # milestones days apart (5, 6): pull the two numbers apart sideways
        dx = 0
        if i > 1 and (_d(m["date"]) - _d(ms[i - 2]["date"])).days < 4:
            dx = 3
        elif i < len(ms) and (_d(ms[i]["date"]) - _d(m["date"])).days < 4:
            dx = -3
        axes[0].annotate(str(i), (_d(m["date"]), 1.0), xycoords=("data", "axes fraction"),
                         xytext=(dx, 2), textcoords="offset points", ha="center",
                         va="bottom", fontsize=_f(8, 6.5), color="0.35")
    key = [f"{i}: {m['label']}" if PAPER else f"{i}: {m['date']}  {m['label']}"
           for i, m in enumerate(ms, 1)]
    half = (len(key) + 1) // 2
    for col, chunk in enumerate((key[:half], key[half:])):
        fig.text(0.02 + 0.5 * col, 0.005, "\n".join(chunk), fontsize=_f(8, 6),
                 color="0.3", va="bottom", ha="left")
    if not PAPER:
        fig.suptitle(
            "Backend port of galpy's test suite: tests passing under each backend",
            fontsize=10, y=0.985,
        )
    loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
    axes[2].xaxis.set_major_locator(loc)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.subplots_adjust(left=0.11 if PAPER else 0.14, right=0.97, top=0.97 if PAPER else 0.93,
                        bottom=0.18 if PAPER else 0.17)
    return fig


_N_BACKEND_MODULES = 88


if __name__ == "__main__":
    make()
