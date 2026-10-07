"""``--Ns``: measure extra N points and MERGE them into an existing measurement JSON.

Used by the speed-measure scripts (shared orbit speed, g12, gV2, gD3) so a later
session (e.g. a larger GPU) can add points at larger N without re-measuring the
rest. The existing JSON keeps all its other points and its own env stamps; each
``--Ns`` run appends one record to its ``extra_points`` list (when, where, with
what versions, which N, and the values it replaced).
"""

import datetime
import json
import math
import os
import socket
import sys


def parse_Ns(s):
    """'1e6,3e6' / '1000000,3000000' -> [1000000, 3000000] (sorted, unique)."""
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if not (math.isfinite(v) and v >= 1 and v == int(v)):
            raise ValueError(f"--Ns: {tok!r} is not a positive integer")
        out.append(int(v))
    if not out:
        raise ValueError("--Ns: no N given")
    return sorted(set(out))


def note(Ns, env):
    """The ``extra_points`` record of this run (``env``: the run's own stamps)."""
    return {
        "added": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "argv": [os.path.basename(sys.argv[0])] + sys.argv[1:],
        "Ns": list(Ns),
        "env": env,
    }


def load(path, required):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    if required:
        sys.exit(f"--Ns merges into an existing JSON, but {path} does not exist")
    return None


def check_same(old, new, what, path):
    """Refuse to mix devices in one series: ``old``/``new`` stamps must agree."""
    if old is not None and new is not None and old != new:
        sys.exit(f"--Ns: {path} was measured on {what} {old!r}, this run is on {new!r}; "
                 "refusing to merge points from different devices into one series")


def merge_results(base, out, nt):
    """Orbit-speed JSONs (``results``: list of {"N": ...}): ``out``'s results
    add to / replace ``base``'s by N; ``base``'s stamps are kept and ``nt`` (this
    run's ``extra_points`` record) is appended, with ``out['stopped_at']``."""
    new = {r["N"]: r for r in out["results"]}
    rec = dict(nt, Ns_measured=sorted(new))
    if "stopped_at" in out:
        rec["stopped_at"] = out["stopped_at"]
    if base is None:
        merged = {k: v for k, v in out.items() if k != "stopped_at"}
        merged["extra_points"] = [rec]
        return merged
    rec["replaced"] = [r for r in base["results"] if r["N"] in new]
    merged = dict(base)
    merged["results"] = sorted(
        [r for r in base["results"] if r["N"] not in new] + list(new.values()),
        key=lambda r: r["N"],
    )
    merged["extra_points"] = list(base.get("extra_points", [])) + [rec]
    return merged


def regrid(old_Ns, vals, grid):
    """Series ``vals`` on ``old_Ns`` -> on ``grid`` (NaN where ``old_Ns`` has no
    point); a short series (e.g. a compile-time list) is NaN-padded first."""
    vals = list(vals) + [float("nan")] * (len(old_Ns) - len(vals))
    at = dict(zip(old_Ns, vals))
    return [at.get(N, float("nan")) for N in grid]


def put(series, grid, Ns, vals):
    """Set ``series`` (on ``grid``) at ``Ns[i]`` to ``vals[i]`` (in place)."""
    for N, v in zip(Ns, vals):
        series[grid.index(N)] = v
    return series
