"""Diagnostic (not figure code): why do g17's large-N GPU points OOM on an 80 GB H100
when the 12 GB TITAN Xp ran them?  python g17_gpu_oom.py torch|jax|seq
  torch: torch (GPU) at N=1e5,3e5,1e6 in a fresh process
  jax  : eager jax (GPU) at N=3e5,1e6 in a fresh process
  seq  : g17's in-process order (jax GPU jit 1e5, jax GPU eager 3e5) then torch 1e5,
         printing what jax holds on the device before torch runs"""
import sys, time
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, torch
torch.set_default_dtype(torch.float64)
from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import MWPotential2014
from paper.figures.g17_staeckel_actionangle_speed.make import _grid, _med
aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
gpu = [d for d in jax.devices() if d.platform == "gpu"][0]
GB = 2**30
def jaxmem():
    s = gpu.memory_stats() or {}
    return f"jax in_use {s.get('bytes_in_use',0)/GB:.1f} GB, peak {s.get('peak_bytes_in_use',0)/GB:.1f} GB, pool {s.get('pool_bytes',s.get('bytes_reserved',0))/GB:.1f} GB"
def torch_at(N):
    torch.cuda.reset_peak_memory_stats()
    try:
        g = [torch.as_tensor(b, device="cuda") for b in _grid(N)]
        aF.actionsFreqsAngles(*g); torch.cuda.synchronize()
        t = _med(lambda: aF.actionsFreqsAngles(*g), warmup=1, repeats=3, sync="torch")
        print(f"torch N={N}: {t:.3f} s, peak {torch.cuda.max_memory_allocated()/GB:.1f} GB", flush=True)
    except Exception as e:
        print(f"torch N={N}: FAIL {str(e)[:160]}", flush=True)
    torch.cuda.empty_cache()
def jax_at(N, jit=False):
    try:
        g = [jax.device_put(jnp.asarray(b), gpu) for b in _grid(N)]
        f = jax.jit(lambda *a: aF.actionsFreqsAngles(*a)) if jit else aF.actionsFreqsAngles
        jax.block_until_ready(f(*g))
        t = _med(lambda: f(*g), warmup=1, repeats=3, sync="jax")
        print(f"jax{' jit' if jit else ''} N={N}: {t:.3f} s; {jaxmem()}", flush=True)
    except Exception as e:
        print(f"jax{' jit' if jit else ''} N={N}: FAIL {str(e)[:160]}; {jaxmem()}", flush=True)
mode = sys.argv[1]
print("mode", mode, torch.cuda.get_device_name(0), flush=True)
if mode == "torch":
    for N in (100000, 300000, 1000000): torch_at(N)
elif mode == "jax":
    for N in (300000, 1000000): jax_at(N)
else:
    jax_at(100000, jit=True); jax_at(300000)
    print("before torch:", jaxmem(), flush=True)
    torch_at(100000)
