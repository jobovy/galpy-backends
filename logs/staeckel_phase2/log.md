# Staeckel Phase-2: freq gradients + action Hessians (PR #1059)

## Approach
Scoped via a prototype workflow (wf_9758b2b3). Both prototype agents independently
found the KEY INSIGHT, overturning the design doc's "needs new second-derivative
integrands (research-grade, L-effort)" framing.

## Dead-end (the design framing)
The doc said dOmega/dx + d2J/dx2 need new analytic second-derivative integrands with
a stronger-than-t2 substitution / integrate-by-parts to tame the S^-3/2 (~1/t2)
turning-point singularity. FALSE. The t2 substitution was never the problem.

## Result (the real cause + fix)
The turning points umin/umax/vmin come from bisect_root (piecewise-constant xp.where)
-> AD gradient is a bracket ARTIFACT, not the implicit derivative. Phase 1 (#1047)
stop_gradient'd the panel limits (legal for the action's 1st deriv: sqrt(S)*du_x
boundary term = 0 at S=0), which DELETES the turning-point-motion boundary term that
CANCELS the S^-3/2 divergence in the 2nd derivative.
FIX (~15 lines, PR #1059): _refine_tp = one differentiable Newton step off the frozen
root (u = sg(u) - S/S_u, theta live) gives du/dtheta = -S_theta/S_u; drop the
stop_gradient on _staeckel_t2_action limits. numpy short-circuit -> byte-identical.
Validated jax+torch: dOr/dR=-1.93268 (was 104.89); Hessian [[0.592,0.305],[0.305,0.684]]
(was [[-278,...]]). Full grad suite 70 passed.

## Gotchas
- The prototype agents' scratch scripts imported galpy from the editable shadow
  (galpy-anotherone), so their "end-to-end validated" claim was unverified -- I
  re-verified independently against the real module (the insight was right, the
  scripts weren't run against the worktree).
- FIRST attempt broke numpy byte-identity: graft_gradient(u,donor)=u+donor-donor is
  only ~machine-eps identity, and the 1/sqrt(S) freq panels amplify ~1e-16 -> ~1e-13.
  Cure: short-circuit _refine_tp under numpy (no autodiff needed there).
- Stale .so trap: copied libgalpy from /var/tmp/pdf2 (pre-#1051) -> 51 phantom
  c=True test failures; rebuild in-worktree.

## Follow-up
c=True C-native freq/angle Jacobian: extend the C actionsJac (#1051) with the same
implicit dumin/dtheta = -S_theta(umin)/S_u(umin) at the solved turning point.
