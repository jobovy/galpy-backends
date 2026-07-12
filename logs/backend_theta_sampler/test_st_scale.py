"""StreamTrack tp_scale mode: normalized concrete grid + traced physical scale.
Query at physical tp under jit; check value + gradient (scale flows)."""
import sys; sys.path.insert(0, "/var/tmp/cint-param")
import numpy
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.df.streamTrack import StreamTrack

u_grid = numpy.linspace(0.0, 1.0, 60)  # CONCRETE normalized axis


def query(scale, tp_phys):
    # traced track values that also depend on `scale` (so grad flows two ways)
    tx = jnp.stack([jnp.sin(2.0 * jnp.pi * u_grid) * scale,
                    jnp.cos(2.0 * jnp.pi * u_grid), u_grid * 0.1], axis=-1)
    tv = jnp.stack([u_grid * 0.0, u_grid * 0.0, u_grid * 0.0], axis=-1)
    tr = StreamTrack(u_grid, tx, tv, tp_scale=scale)  # physical tp = u * scale
    return tr.x(tp_phys, use_physical=False)


scale0 = jnp.asarray(3.0)
tp_phys = 1.5  # physical query (inside [0, scale0=3])
print("=== StreamTrack tp_scale under jit ===")
v = jax.jit(lambda s: query(s, tp_phys))(scale0)
print("  x(tp=1.5) =", float(v))
# reference: physical tp=1.5 -> u=1.5/3=0.5 -> sin(2pi*0.5)*3 = 0
print("  expected ~ sin(2pi*0.5)*3 =", float(numpy.sin(2*numpy.pi*0.5)*3))
# gradient wrt scale (flows through both the values and the tp->u mapping)
g = float(jax.jit(jax.grad(lambda s: query(s, tp_phys)))(scale0))
h = 1e-6
gfd = (float(jax.jit(lambda s: query(s, tp_phys))(scale0 + h)) -
       float(jax.jit(lambda s: query(s, tp_phys))(scale0 - h))) / (2 * h)
print(f"  d(x)/d(scale): AD={g:.6e} FD={gfd:.6e} REL={abs(g-gfd)/max(abs(g),1e-9):.2e}")
print("  tp_grid()[-1] (physical) =", float(jax.jit(lambda s: StreamTrack(u_grid, jnp.stack([u_grid,u_grid,u_grid],-1)*s, jnp.stack([u_grid,u_grid,u_grid],-1), tp_scale=s).tp_grid()[-1])(scale0)))
