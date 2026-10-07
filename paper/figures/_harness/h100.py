"""H100 series next to the TITAN Xp ones.

The speed figures were measured on the dev box's NVIDIA TITAN Xp (a consumer card,
float64 at 1/32 of float32) and again on an NVIDIA H100 (Trillium; see
tools/H100_SESSION.md). The H100 measurement of a figure lives in
``<figure>/data/h100/<same file>.json``; its GPU series are drawn as extra lines
with the TITAN line's colour and hollow markers, labelled "H100".
"""

import json
import os

TITAN = "TITAN Xp"
H100 = "H100"


def load(figdir, fname):
    """The H100 measurement of ``<figdir>/data/<fname>``, or None."""
    path = os.path.join(figdir, "data", "h100", fname)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def merge(main, h100, containers, is_gpu=lambda name: "GPU" in name, rename=None):
    """Copy the GPU series of ``h100[container]`` into ``main[container]``.

    A series ``name`` becomes ``rename(name)`` (default: "GPU" -> "H100");
    main's own GPU series keep their names. Returns the list of new names.
    """
    rename = rename or (lambda n: n.replace("GPU", H100))
    new = []
    if not h100:
        return new
    for c in containers:
        for name, val in (h100.get(c) or {}).items():
            if is_gpu(name):
                main.setdefault(c, {})[rename(name)] = val
                new.append(rename(name))
    return new


def is_h100(name):
    return H100 in name


def title_gpu(name, have_h100):
    """Label for a TITAN-measured GPU series once H100 lines are present."""
    return name.replace("GPU", TITAN) if have_h100 and "GPU" in name else name
