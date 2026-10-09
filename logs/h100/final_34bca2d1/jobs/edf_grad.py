"""Print the jax autodiff and FD values of test_moment_grad_through_evolution_vs_fd
(meanvT, vertexdev) exactly as the test computes them; one fresh process per call.
Diagnostic only: imports the test module's helpers, changes no galpy code."""
import os, sys
sys.path.insert(0, os.path.join(os.environ["GALPY_WORKTREE"], "tests"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import test_backend_evolveddiskdf as T
from galpy.backend import use, as_numpy
edf = T._make_edf(); R0, h = T._R, 1e-5
print("device", jax.devices()[0], "XLA_FLAGS=" + os.environ.get("XLA_FLAGS", ""), flush=True)
for moment in ("meanvT", "vertexdev"):
    fn = T._MOMENTS[moment]
    def val(R):
        with use("jax", force=True):
            return as_numpy(jnp.asarray(fn(edf, R)).reshape(()))
    gfd = (float(val(T._scalar("jax", R0 + h))) - float(val(T._scalar("jax", R0 - h)))) / (2 * h)
    with use("jax", force=True):
        g = float(jax.grad(lambda R: jnp.asarray(fn(edf, R)).reshape(()))(jnp.asarray(R0)))
    print(f"{moment}: grad {g!r}  fd {gfd!r}  rel {abs(g-gfd)/abs(gfd):.3e}  pass(rtol 2e-4) {abs(g-gfd) <= 1e-7 + 2e-4*abs(gfd)}", flush=True)
