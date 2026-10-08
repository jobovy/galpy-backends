"""Loop-aware FLOP / byte census of a compiled XLA (CPU) module.

XLA's ``compiled.cost_analysis()`` counts every while-loop body ONCE (the trip
count is ignored; checked: a fori_loop of 1, 10 and 100 iterations all report the
same flops). The kernels this note analyses spend most of their work inside
loops (jax's native igamma = two convergence while-loops; diffrax's step loop;
galpy's fixed-iteration Newton/bisection), so this module re-derives the census
from the optimized HLO text with a per-while trip count:

  * flops         -- elementwise ops x output elements, XLA's convention (add,
                     mul, div, compare, select, logic, ... each = 1), split by
                     category so a reader can re-weight;
  * transcendentals -- exp, log, pow, sqrt, rsqrt, atan2, sin, cos, ... (XLA's
                     convention: counted separately from flops);
  * bytes         -- for every top-level instruction of a non-fused computation
                     (entry, loop bodies/conditions): output bytes + operand
                     bytes (fusion = one kernel: its operands in, its output out),
                     the same model as XLA's "bytes accessed".

With every trip count set to 1 the totals reproduce ``cost_analysis()`` (checked
by ``crosscheck`` in each driver), which validates the parser; the trip counts
are then supplied per loop by the caller (``trips(while_info) -> int``).
"""

import re

_DT_BYTES = {
    "pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "f16": 2, "bf16": 2,
    "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8, "c64": 8, "c128": 16,
}
_TRANSC = {
    "exponential", "exponential-minus-one", "log", "log-plus-one", "power", "sqrt",
    "rsqrt", "cbrt", "tanh", "logistic", "sine", "cosine", "tan", "atan2", "erf",
    "erf-inv",
}
# no arithmetic (layout / data movement / bookkeeping)
_FREE = {
    "parameter", "constant", "broadcast", "reshape", "bitcast", "copy", "transpose",
    "slice", "concatenate", "dynamic-slice", "dynamic-update-slice",
    "get-tuple-element", "tuple", "iota", "pad", "gather", "scatter", "reverse",
    "after-all", "partition-id", "replica-id", "copy-start", "copy-done",
    "bitcast-convert", "convert", "opt-barrier", "add-dependency",
}
_CMP = {"compare", "select", "and", "or", "not", "xor", "is-finite", "clamp"}
_INT_SHIFT = {"shift-left", "shift-right-logical", "shift-right-arithmetic", "popcnt"}


def _split_top(s):
    """Split a comma-separated list at depth 0 of (), {}, []."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur and "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _balanced(s, i):
    """Index just past the bracket group starting at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] in "([{":
            depth += 1
        elif s[j] in ")]}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError(s)


_ARR = re.compile(r"(pred|[subfc]\d+|bf16)\[([^\]]*)\]")


def shape_info(sh):
    """(list of (dtype, elements) per leaf, total bytes) of an HLO shape string."""
    leaves = []
    for m in _ARR.finditer(re.sub(r"/\*.*?\*/", "", sh)):
        dt, dims = m.group(1), m.group(2)
        n = 1
        for d in dims.split(","):
            d = d.strip()
            if d:
                n *= int(d)
        leaves.append((dt, n))
    return leaves, sum(_DT_BYTES.get(dt, 8) * n for dt, n in leaves)


def parse(text):
    comps, cur, entry = {}, None, None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^(ENTRY\s+)?%([\w.\-]+)\s*\(.*\)\s*->\s*.*\{$", line)
        if m:
            cur = m.group(2)
            comps[cur] = []
            if m.group(1):
                entry = cur
            continue
        if line == "}":
            cur = None
            continue
        if cur is None:
            continue
        m = re.match(r"^\s*(ROOT\s+)?%([\w.\-]+)\s*=\s*(.*)$", line)
        if not m:
            continue
        name, rest = m.group(2), m.group(3)
        if rest.startswith("("):
            j = _balanced(rest, 0)
            shape, rest2 = rest[:j], rest[j:].lstrip()
        else:
            shape, _, rest2 = rest.partition(" ")
        mo = re.match(r"([\w\-]+)\(", rest2)
        if not mo:
            continue
        opcode = mo.group(1)
        k = mo.end() - 1
        j = _balanced(rest2, k)
        operands = [o.split()[-1].lstrip("%") for o in _split_top(rest2[k + 1 : j - 1]) if o]
        attrs = rest2[j:]
        comps[cur].append(dict(name=name, shape=shape, opcode=opcode, operands=operands,
                               attrs=attrs, root=bool(m.group(1))))
    return comps, entry


def _attr_comp(attrs, key):
    m = re.search(key + r"=%([\w.\-]+)", attrs)
    return m.group(1) if m else None


def known_trip(attrs):
    m = re.search(r'"known_trip_count":\{"n":"(\d+)"\}', attrs)
    return int(m.group(1)) if m else None


class Census:
    """Recursive loop-aware census. ``trips(info)`` gets a dict with the while's
    name, carry shape, number of carry leaves, the known trip count (or None),
    and the nesting path, and returns the number of BODY executions."""

    def __init__(self, text, trips=None, ideal=False):
        """ideal=True: the perfect-fusion LOWER bound on bytes -- each loop body is
        one kernel that reads and writes its carry once per iteration (plus the
        loop predicate), and the entry computation moves only its parameters and
        result; all other intermediates stay on chip."""
        self.ideal = ideal
        self.comps, self.entry = parse(text)
        self.trips = trips or (lambda info: info["known"] or 1)
        self.whiles = []  # (path, info, trips) for reporting

    def _zero(self):
        return dict(arith=0.0, div=0.0, cmp=0.0, int=0.0, other=0.0, transc=0.0,
                    bytes=0.0, transc_by={}, launches=0.0)

    @staticmethod
    def _add(a, b, w=1.0):
        for k in ("arith", "div", "cmp", "int", "other", "transc", "bytes", "launches"):
            a[k] += w * b[k]
        for k, v in b["transc_by"].items():
            a["transc_by"][k] = a["transc_by"].get(k, 0.0) + w * v

    def _elementwise(self, ins, acc):
        op = ins["opcode"]
        leaves, _ = shape_info(ins["shape"])
        n = sum(e for _, e in leaves)
        dt = leaves[0][0] if leaves else "f64"
        if op in _FREE:
            return
        if op in _TRANSC:
            acc["transc"] += n
            acc["transc_by"][op] = acc["transc_by"].get(op, 0.0) + n
        elif op in _CMP:
            acc["cmp"] += n
        elif op in _INT_SHIFT or dt[0] in "su" and dt != "pred":
            acc["int"] += n
        elif op == "divide" or op == "remainder":
            acc["div"] += n
        elif op in ("add", "subtract", "multiply", "negate", "abs", "maximum", "minimum",
                    "sign", "floor", "ceil", "round-nearest-even", "round-nearest-afz"):
            acc["arith"] += n
        elif op in ("reduce", "reduce-window"):
            # one combiner application per input element
            ops = ins["operands"]
            acc["other"] += n  # approximated below by the caller with input size
        else:
            acc["other"] += n

    def cost(self, comp, fused=False, path=()):
        acc = self._zero()
        syms = {}
        for ins in self.comps[comp]:
            syms[ins["name"]] = ins["shape"]
        for ins in self.comps[comp]:
            op = ins["opcode"]
            if op == "fusion":
                sub = self.cost(_attr_comp(ins["attrs"], "calls"), fused=True, path=path)
                self._add(acc, sub)
            elif op == "while":
                body = _attr_comp(ins["attrs"], "body")
                cond = _attr_comp(ins["attrs"], "condition")
                leaves, _ = shape_info(ins["shape"])
                info = dict(name=ins["name"], shape=ins["shape"], nleaves=len(leaves),
                            known=known_trip(ins["attrs"]), path=path,
                            lane=max((e for _, e in leaves), default=1))
                n = self.trips(info)
                self.whiles.append((path, info, n))
                b = self.cost(body, path=path + (ins["name"],))
                c = self.cost(cond, path=path + (ins["name"],))
                self._add(acc, b, n)
                self._add(acc, c, n + 1)
                if self.ideal:
                    _, cb = shape_info(ins["shape"])
                    acc["bytes"] += n * 2 * cb + (n + 1) * info["lane"]  # carry in+out, pred
                continue  # the loop's traffic is its body's
            elif op in ("call", "async-start"):
                sub = _attr_comp(ins["attrs"], "to_apply") or _attr_comp(ins["attrs"], "calls")
                if sub:
                    self._add(acc, self.cost(sub, fused=fused, path=path))
                continue
            elif op == "conditional":
                m = re.search(r"branch_computations=\{([^}]*)\}", ins["attrs"])
                names = ([s.strip().lstrip("%") for s in m.group(1).split(",")] if m else
                         [_attr_comp(ins["attrs"], k) for k in ("true_computation",
                                                                 "false_computation")])
                # cost of the most expensive branch (one runs)
                best = None
                for nm in names:
                    if nm:
                        cst = self.cost(nm, path=path)
                        if best is None or cst["bytes"] + cst["arith"] > best["bytes"] + best["arith"]:
                            best = cst
                if best:
                    self._add(acc, best)
            elif op in ("reduce", "reduce-window", "all-reduce"):
                # one combiner op per INPUT element
                nin = 0
                for o in ins["operands"][: max(1, len(ins["operands"]) // 2)]:
                    if o in syms:
                        nin += sum(e for _, e in shape_info(syms[o])[0])
                leaves, _ = shape_info(ins["shape"])
                dt = leaves[0][0] if leaves else "f64"
                acc["cmp" if dt == "pred" else "arith"] += nin
            elif op == "dot":
                leaves, _ = shape_info(ins["shape"])
                nout = sum(e for _, e in leaves)
                a = ins["operands"][0]
                ka = shape_info(syms.get(a, "f64[1]"))[0]
                m = re.search(r"lhs_contracting_dims=\{([\d,]*)\}", ins["attrs"])
                dims = re.search(r"\[([^\]]*)\]", syms.get(a, "f64[1]")).group(1).split(",")
                k = 1
                if m and m.group(1):
                    for d in m.group(1).split(","):
                        k *= int(dims[int(d)])
                acc["arith"] += 2.0 * nout * k
            else:
                self._elementwise(ins, acc)
            if not fused and op not in ("parameter", "constant", "get-tuple-element",
                                        "tuple", "bitcast", "after-all"):
                if not self.ideal:
                    acc["bytes"] += self._io_bytes(ins, syms)
                elif comp == self.entry and (ins["root"] or op == "parameter"):
                    acc["bytes"] += shape_info(ins["shape"])[1]
                acc["launches"] += 1
            if self.ideal and comp == self.entry and op == "parameter":
                acc["bytes"] += shape_info(ins["shape"])[1]
        return acc

    def _io_bytes(self, ins, syms):
        """Output + operand bytes, except that an in-place dynamic-update-slice
        (top level, or the ROOT of a fusion) moves only its update: its buffer
        operand is aliased to the output and not re-streamed."""
        ob = shape_info(ins["shape"])[1]
        ops = [o for o in ins["operands"] if o in syms]
        sizes = [shape_info(syms[o])[1] for o in ops]
        if ins["opcode"] == "dynamic-update-slice" and len(sizes) >= 2:
            return 2 * sizes[1] + sum(sizes[2:])
        if ins["opcode"] == "fusion":
            comp = _attr_comp(ins["attrs"], "calls")
            body = self.comps.get(comp, [])
            root = next((i for i in body if i["root"]), None)
            if root is not None and root["opcode"] == "dynamic-update-slice":
                fsyms = {i["name"]: i for i in body}
                buf = fsyms.get(root["operands"][0])
                upd = fsyms.get(root["operands"][1])
                if buf is not None and buf["opcode"] == "parameter" and upd is not None:
                    k = int(buf["operands"][0])
                    if k < len(sizes):
                        return sum(sizes) - sizes[k] + shape_info(upd["shape"])[1]
        return ob + sum(sizes)

    def total(self):
        self.whiles = []
        return self.cost(self.entry)


def flops(acc):
    """XLA-convention flops (everything elementwise except transcendentals)."""
    return acc["arith"] + acc["div"] + acc["cmp"] + acc["int"] + acc["other"]


def summary(acc, per=1.0, w_transc=20.0):
    f = flops(acc) / per
    t = acc["transc"] / per
    return dict(
        flops_xla=f,
        transcendentals=t,
        flop_equiv=f + w_transc * t,
        arith=acc["arith"] / per,
        div=acc["div"] / per,
        cmp_select_logic=acc["cmp"] / per,
        integer=acc["int"] / per,
        other=acc["other"] / per,
        transc_by={k: v / per for k, v in sorted(acc["transc_by"].items())},
        bytes=acc["bytes"] / per,
        kernels=acc["launches"],
    )
