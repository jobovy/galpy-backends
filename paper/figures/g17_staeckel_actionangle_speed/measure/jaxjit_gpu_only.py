"""Re-measure g17's jax (GPU, jit) series ALONE, through make.py's own child runner
(_run_series: same grid, timing helper and compile-excluded warm calls), on a denser
N grid, and splice it into the figure's JSON.

Usage: PYTHONPATH=<galpy worktree> python jaxjit_gpu_only.py [path/to/g17.json]
(default: data/g17_staeckel_actionangle_speed.json; the H100 copy is
data/h100/g17_staeckel_actionangle_speed.json). Written for the device-aware
incomplete-gamma switch (galpy branch backend/special-gpu-gate): a staged call on
a GPU takes galpy's series at every size, so only this series changes; the
galpy commit it was measured at is stamped in galpy_commit_parts["gpu_jaxjit"]."""
import datetime, json, os, subprocess, sys, importlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
m = importlib.import_module("paper.figures.g17_staeckel_actionangle_speed.make")  # noqa: E402
NAME = "jax (GPU, jit)"
NS = [1, 100, 1000, 3000, 10000, 30000, 100000]  # < 1e6: compile-time OOM risk
JSON = sys.argv[1] if len(sys.argv) > 1 else m._JSON
times, status, cenv = m._run_series(NAME, NS)
print(f"[{NAME}] {status}: {times}", flush=True)
if status != "ok":
    raise SystemExit("not written: " + status)
wt = os.environ["PYTHONPATH"].split(os.pathsep)[0]
sha = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD"], capture_output=True,
                     text=True, check=True).stdout.strip()
patch = {"Ns_jaxjit_gpu": NS, "times": times, "status": status, "cenv": cenv,
         "commit": {"sha": sha, "worktree": wt,
                    "date": datetime.datetime.now().isoformat(timespec="seconds"),
                    "XLA_FLAGS": os.environ.get("XLA_FLAGS", "unset"),
                    "note": "jax (GPU, jit) re-measured alone (jaxjit_gpu_only.py) with "
                            "the device-aware incomplete-gamma switch"}}
print("PATCH " + json.dumps(patch), flush=True)  # what a remote run sends back
d = json.load(open(JSON))
d["Ns_jaxjit_gpu"] = patch["Ns_jaxjit_gpu"]
d["series"][NAME] = patch["times"]
d["status"][NAME] = patch["status"]
d["env"].setdefault("series", {})[NAME] = patch["cenv"]
d.setdefault("galpy_commit_parts", {})["gpu_jaxjit"] = patch["commit"]
json.dump(d, open(JSON, "w"), indent=2)
print("written", JSON)
