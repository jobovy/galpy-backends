# Vacation status snapshot — 2026-08-25 (Codex review of #1204: findings 1/2/3/4/6)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except (a) the main->develop rebase (parked) and
## (b) review finding 5, which needs a direction call (see below).

## Branch state

    feat/backends          982b1781
    feat/backends-develop  eabc3a5a   (last merge: gh#1380 ledger Group-1)

## Open PRs (2 = my cap)

* **gh#1381** `fix/gammainc-endpoint-grad` @ cc8286a3 — finding 1: d/dx gammainc
  was NaN at the x=0 endpoint, plus the int-dtype follow-up (`full_like` on an
  INTEGER order raises on torch; EinastoPotential passes one).
  CI: 95 green / 53 queued. Reds are **all external or settled**:
  `test_from_name_*` = the SIMBAD/VizieR outage (service re-probed 2026-08-25
  and it is BACK; failed run re-run), the two macos reds are flakes proven by
  the same-SHA push-vs-pull_request A/B, `codecov/project` is the incomplete-
  upload phantom while backend shards are still queued.
* **gh#1382** `fix/quadrature-batching-anyaxisym-units` @ (head) — findings
  2 + 10 + the torch `.to()` construction bug. CI: 89 green / 64 queued, **zero
  failures**.

## Committed locally, PR HELD behind the two-PR cap

* **`fix/coords-inverse-transforms`** @ e999b04c (worktree `/var/tmp/rev1`) —
  findings 3 + 4 + 6, stacked linearly on eabc3a5a. **finding 6 is now 7 of 7.**
  Contains, beyond the migration itself, three defects the migration exposed:
  1. **f32 defect in ALREADY-MERGED code.** `galcen_extra_rot` is a float64
     constant; torch refuses f64@f32, so `galcenrect_to_XYZ` /
     `galcenrect_to_vxvyvz` / `XYZ_to_galcenrect` raised for ANY torch user at
     torch's default dtype. Invisible because the suite is all-float64.
     `_promote_pair` casts via `result_type` = numpy's own promotion.
  2. **My own dead branch.** `XYZ_to_galcenrect` had a backend branch but no
     `@backendNative`, so `scalarDecorator`'s numpy ndim probe raised before
     reaching it on exactly the grad/traced inputs it was for.
  3. **Resolved-namespace / uncoerced-input** in `lbd_to_XYZ`,
     `vrpmllpmbb_to_vxvyvz`, `galcencyl_to_galcenrect`, `lambdanu_to_Rz`.
  Gate: `tests/test_coords.py` **98/98 under numpy, torch AND jax** (torch was
  10 failures before, 0 after; the pre-stack base is also 0, so all 10 were
  mine). numpy byte-identity verified WITH a 1-ulp negative control.
  Gradients grad-vs-FD 1e-8..1e-11 at `Zsun != 0` where they are non-vacuous.
  Wider shard runs (test_quantity+test_coords, and the conversion/util/
  kuzminkutuzov shard) in all 3 modes were in flight at end of wake.

  NOTE: the older branch `fix/traced-surfdens-and-galcen-rot` @ be5ff216 is a
  DIVERGENT earlier copy of the same work (same subjects, different SHAs).
  `fix/coords-inverse-transforms` supersedes it — open that one, not both.

* **`fix/map-coordinates-grid-dispatch`** (worktree `/var/tmp/interpfix`) —
  finding 7.

## Open review findings not yet done

* **finding 5** — NEEDS JO'S CALL. Jo steered "`set_jit("jax")` only makes
  sense when working on the backend, so implies `--backend`", but the fork
  remains: (a) `set_jit` turns the backend on, or (b) it REQUIRES one already
  set and raises otherwise. I lean (b). Not implemented.
* **findings 8 & 9** — traced constructor params (`min(a, rc)`,
  TwoPowerTriaxial branches, constantbetaPowerLawdf asserts) and `hyp2f1`
  scalar params. All four finding-8 sites reproduce `TracerBoolConversionError`.
  Needs a per-site structural-vs-data decision.
* **finding 11 (mine, reported not fixed)** — no torch gradient through
  surfdens, ever.

## Housekeeping

* Cron `5451d57e` (hourly at :13) verified alive 2026-08-25. Session-only and
  expires ~7d after creation — re-create on ANY wake where CronList is empty.
* `/dev/sda6` (root, carries /tmp and /var/tmp) hit **100% full** this wake and
  wedged every command with ENOSPC. Freed ~10G by deleting per-process
  `torchinductor_galpy_*` caches + pip/uv caches. Those caches are never GC'd
  and leak one directory per forced-torch test process — expect this to recur.
