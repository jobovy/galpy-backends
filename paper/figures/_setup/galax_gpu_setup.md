# galax GPU setup (TITAN Xp, sm_61) — working recipe

Reproducible setup for the galpy-vs-galax orbit-integration speed benchmark on
the TITAN Xp (compute capability 6.1 / sm_61, driver 580.159.03, 12 GiB).

## The trap (why this was dependency hell)

1. **PyPI galax is ancient (0.0.2)** and incompatible with modern equinox
   (`ImportError: _has_dataclass_init`). Must install the GitHub dev version.
2. The GalacticDynamics ecosystem (galax/unxt/coordinax/quaxed/quax/plum)
   co-evolves; ad-hoc `pip install` re-resolves and downgrades members into
   mutually-incompatible versions (`TypedInt -> jax.Array` failures, quax
   `stage_value "must override"`). **Use galax's own `uv.lock`** to get the
   exact tested set, and add CUDA with `--no-deps` so it can't perturb it.
3. **galax's lock pins jax/jaxlib 0.8.0**, NOT the 0.10.1 that's in the main
   `py313` galpy env. Installing the jax-cuda12 **0.10.1** plugin against
   jaxlib **0.8.0** gives a callback API skew:
   `AttributeError: module 'jaxlib.xla_client' has no attribute
   'register_custom_type_handler'` → `No FFI handler registered for
   xla_ffi_python_gpu_callback on platform CUDA`. The CUDA plugin version MUST
   match jaxlib.
4. CUDA **13** dropped Pascal (sm_61); use the **cu12** plugin + nvidia-*-cu12
   wheels (CUDA 12.9 works on sm_61).

## Working recipe

```bash
# 1. Tools
python3 -m venv /var/tmp/galax-venv && /var/tmp/galax-venv/bin/pip install uv
UV=/var/tmp/galax-venv/bin/uv

# 2. galax dev + its exact locked deps (jax/jaxlib 0.8.0, CPU)
git clone --depth 1 https://github.com/GalacticDynamics/galax.git /var/tmp/galax-src
cd /var/tmp/galax-src
PIP_CACHE_DIR=/var/tmp/pip-cache UV_CACHE_DIR=/var/tmp/uv-cache \
  $UV sync --frozen --no-dev          # -> .venv with the locked, working set

# 3. Add the CUDA-12 plugin MATCHING jaxlib (0.8.0), plus nvidia cu12 wheels,
#    with --no-deps so the locked galax ecosystem is NOT re-resolved.
VIRTUAL_ENV=/var/tmp/galax-src/.venv PIP_CACHE_DIR=/var/tmp/pip-cache UV_CACHE_DIR=/var/tmp/uv-cache \
  $UV pip install --no-deps \
    jax-cuda12-plugin==0.8.0 jax-cuda12-pjrt==0.8.0 \
    nvidia-cublas-cu12 nvidia-cuda-cccl-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12

# 4. Verify
/var/tmp/galax-src/.venv/bin/python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0)]
```

## Working version set (uv pip list)

```
galax              0.1.dev1+g9540f5830   (GitHub main @ 9540f5830)
jax                0.8.0
jaxlib             0.8.0
jax-cuda12-plugin  0.8.0     <-- MUST match jaxlib
jax-cuda12-pjrt    0.8.0
nvidia-*-cu12      12.9.x    (CUDA 12.9; cu12 supports sm_61, cu13 does NOT)
equinox 0.13.8  diffrax 0.7.0  quax 0.3.2  quaxed 0.10.5
unxt 1.11.2  unxt-api 1.10.1  coordinax 0.23.3  plum-dispatch 2.8.0
```

## API notes

- Use the **modern solver**, not the legacy `gd.evaluate_orbit` (the legacy
  integrator path is fine on CPU but its design is being deprecated):

  ```python
  import galax.potential as gp, galax.dynamics as gd, galax.coordinates as gc
  import unxt as u, jax
  jax.config.update("jax_enable_x64", True)
  mw     = gp.BovyMWPotential2014()          # == MWPotential2014, default params
  field  = gd.fields.HamiltonianField(mw)
  solver = gd.OrbitSolver()                  # diffrax Dopri8
  w0  = gc.PhaseSpaceCoordinate(q=u.Q([[8.,0.,0.1]], "kpc"),
                                p=u.Q([[10.,220.,8.]], "km/s"), t=u.Q(0., "Gyr"))
  soln = solver.solve(field, w0, u.Q(2., "Gyr"))   # runs on GPU
  ```

- With the **matched** 0.8.0 plugin, GPU callbacks work and **no
  `EQX_ON_ERROR` flag is needed**. (If you ever run with a mismatched plugin,
  `EQX_ON_ERROR=nan` disables equinox's callback-based error checks as a
  stopgap, but fix the plugin version instead.)
- jax sees the GPU as `CudaDevice(id=0)`; a `block_until_ready()` is required
  to time GPU work (async dispatch).
