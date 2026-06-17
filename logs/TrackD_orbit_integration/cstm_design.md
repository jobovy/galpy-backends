# C-STM orbit-autodiff wrappers — design brief (2026-06-17)

Design checkpoint for Track D Pjax/Ptorch-orbit (#40). Produced by a design agent
reading feat/backends. Implementation-ready; split into PRs below.

## Key facts (feat/backends)
- C variational integrator already merged: `integrateFullOrbit_dxdv` integrates a
  12-D rect system (6 state + ONE 6-vector deviation), `evalRectDeriv_dxdv`
  (integrateFullOrbit.c:1453), entry :1203 (dim=12). One C call = one deviation.
- Full 6x6 M = integrate the 6 canonical basis vectors, stack as columns — exactly
  `_integrate_stm_3d` (tests/test_orbit.py:768), rectIn=rectOut=True (rect frame).
- In-backend ODE path (#906) exists but is NOT yet routed into Orbit.integrate
  (Orbits.py has zero backend awareness). So routing/dispatch is greenfield.

## Design
- Shared numpy helper `_reference/inbackend_stm.py::c_stm_forward`: cyl->rect IC,
  6 C calls (parse pot ONCE), stack -> (xt_orbit, M_rect (nt,6,6), J0, Jt).
- `_jax/orbit_stm.py`: jax.custom_vjp; fwd via jax.pure_callback (returns xt AND M
  residual; vmap_method="expand_dims"); bwd = J0^T M_rect^T Jt^-T · ct (IC-only).
  nondiff_argnums = pot, ts, method, rtol, atol.
- `_torch/orbit_stm.py`: torch.autograd.Function; forward runs C (detach().cpu()
  for CUDA), saves M/J0/Jt; backward = einsum -> vxvv grad, None for rest. gradcheck.
- IC-only gradients. Params can't pass through ctypes->C and C has no dx/dtheta;
  param gradients stay with the in-backend ODE path. (PR-E future: C param sens.)

## Phasing
- PR-B (overnight, autonomous, NO Orbit changes): shared helper + jax + torch
  functional `integrate_stm()` + full test battery. Reviewable in isolation.
- PR-C (needs user decisions): Orbit.integrate routing — IC capture in __init__,
  dispatch in _integrate_impl:1613, compose with the (also-unrouted) ODE path.
- PR-D (later): single-call C STM shim (dim=42) for speed. PR-E: C param sensitivities.

## Tests (tests/test_backend_orbit_stm.py)
forward==numpy C; jax grad vs FD; jacrev==jacfwd==assembled-M; torch gradcheck;
C-STM vs in-backend-ODE agreement; torch grad==jax grad; det M=1 / M^T Omega M=Omega
on the AD-recovered M; closed-form harmonic + Kepler STM (test_orbit.py:1041,1115);
matrix {rk4_c,rk6_c,dopr54_c,dop853_c} x {MiyamotoNagai, MWPotential2014, non-axi}.

## Open questions for the user (PR-C routing)
1. IC capture: stash self._ic_backend=vxvv in Orbit.__init__ before numpy coercion
   (keeps self.vxvv numpy / API byte-identical)? [recommended]
2. Selection knob when both C-STM and in-backend ODE are eligible: method-string
   overload vs new `grad_method=`/`differentiable=` kwarg?
3. Confirm IC-only param scope for C-STM (params via ODE path). [recommended]
4. Must o.R(t) etc. return backend tensors w/ live grads in PR-1, or is a functional
   integrate_stm(...)->array enough first? [functional first is simpler]
5. Routing ownership: wire the in-backend ODE path into Orbit.integrate FIRST
   (PR-A), or let C-STM introduce the dispatch hook?
6. Public frame: Orbit order (fold cyl<->rect Jacobians) vs rectangular. [Orbit order]
