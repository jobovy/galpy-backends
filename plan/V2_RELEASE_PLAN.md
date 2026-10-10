# v2 branch + 2.0.0b1 pre-release plan

Prepared 2026-10-09 against `feat/backends` b20733558 / `main` 8947afb90. Nothing here
has been pushed, tagged or changed on GitHub. Diffs (verified: `git apply` clean on
those SHAs, YAML parses, sdist builds as `galpy-2.0.0b1.tar.gz`):

| file | applies to | who pushes |
|---|---|---|
| `v2_release/v2-workflows.diff` | v2 (6 workflows: branch rename, pre-release guards, Python 3.10 dropped) | Jo (workflow files) |
| `v2_release/v2-version.diff` | v2 (cfg, version strings, extras, `python_requires>=3.11`, README/docs, HISTORY, version check, comments) | me (PR into v2) |
| `v2_release/main-docs.diff` | main (installation.rst, `"galpy<2"`) | me (PR into main; Jo merges) |

Decisions (Jo): `feat/backends` → `v2`; first pre-release 2.0.0b1 (then bN, rcN, final);
main stays 1.x; main fixes reach v2 by cherry-pick PRs; no rebasing v2 once a tag exists;
**v2 (galpy 2.x) drops Python 3.10** (main/1.x keeps it), see §6.

---

## 1. Branch rename

### Current state
- Rulesets (`gh api repos/jobovy/galpy/rulesets`):
  - **`feat`** (17915683, active): include `refs/heads/feat/backends*`. Rules: deletion,
    required_linear_history, non_fast_forward, pull_request (merge methods merge/squash/rebase),
    required_status_checks `all-clear`. No bypass actors.
  - **`main`** (995570, active): `~DEFAULT_BRANCH`, same rules, squash/rebase only.
  - **`Agent`** (19751035, *disabled*): `~DEFAULT_BRANCH`, `refs/heads/feat/backends`; deletion, non_fast_forward, update.
- `feat/backends-develop` still exists. Its last commit is b06985892 (2026-09-02). It has 366
  commits that are not on feat/backends (orphaned by the rebase), and its last merged PR was #1428. Delete it.
- Open PRs with base feat/backends: #1648, #1659, #1661, #1662. Integration PR #892
  (head feat/backends → main) is still open but will never merge.
- main is 6 commits ahead of feat/backends (merge-base 7c675eb63).

### What GitHub does on rename
- **Automatic:** retargets open PRs whose *base* is the branch. Updates classic branch-protection rules
  and draft releases. Redirects web URLs (`/tree/feat/backends/...`, `/blob/...`) to v2.
  Shows contributors a "branch renamed" notice.
- **Not automatic:**
  - Rulesets match by pattern and keep matching only `feat/backends*`, so **v2 is
    unprotected until the `feat` ruleset is edited.**
  - Workflow `on: push/pull_request: branches:` filters and `github.ref`/`base_ref` checks in `if:`.
  - Codecov, Read the Docs, badges and docs links that name the branch.
  - Local clones, worktrees and gh-stack metadata.
  - git protocol pushes are not redirected: a stale `git push origin feat/backends` *recreates* the old branch.
  - PRs whose *head* is the branch (#892): close it beforehand rather than depend on what GitHub does.

### References to `feat/backends` in the repo (`git grep` on b20733558)
- `.github/workflows/backend-tests.yml` (26 hits): push/PR filters, the job `if:`, and the
  "integration PR (feat/backends → main)" sticky-comment steps.
- `.github/workflows/build.yml` (2 hits): push filter.
- None in README.md, CONTRIBUTING.md, codecov.yml, .readthedocs.yaml, setup.py, pyproject.toml, doc/, scripts, tools/ or galpy/.
- Historical comments in `tests/backend_xfail.txt` (9), `tests/backend_slow_skip.txt` (1),
  `tests/test_backend_{actionAngle,anyaxisymdisk,dtype_policy}.py` (1 each). **Leave these
  unchanged.** They record where past runs happened, and rewriting them would make the log wrong.

`v2_release/v2-workflows.diff` (Jo applies) changes the following:
- `backend-tests.yml`:
  - `feat/backends` → `v2` and `feat/backends-develop` dropped, in the push/PR filters and the job `if:`.
  - The two integration-PR steps are deleted (v2 never merges to main, and #892 will be closed).
- `build.yml`: push filter → `main`, `v2`.
- `wheels.yml`: adds `v2` to the push filter. Today no v2 code has ever gone through
  cibuildwheel (manylinux/macOS/Windows, cp311–cp315 on v2). This makes those builds run before a tag
  instead of first running at release. S3 upload stays gated to `refs/heads/main`.
- `test-doc-notebooks.yml`: adds `v2` to the push filter. PRs into v2 already run it, because its `pull_request` has no branch filter.
- `build_pyodide_wheels.yml`: the release-time S3 upload now requires `!github.event.release.prerelease`.
  Otherwise a 2.0.0b1 release would overwrite the `galpy-latest-*.whl` pyodide wheels that the v1 docs serve.
  A release event runs the workflow files *at the tagged commit*, so this guard has to be on v2. It is.

### Ruleset changes (Jo)
```bash
# (a) protect v2 (and stop protecting the dead feat/backends* names); rename the ruleset
gh api -X PUT repos/jobovy/galpy/rulesets/17915683 --input - <<'EOF'
{"name": "v2",
 "conditions": {"ref_name": {"include": ["refs/heads/v2"], "exclude": []}}}
EOF
# optional, same call: drop "merge" from allowed_merge_methods (linear history already forbids merge commits)

# (b) after the rename: block stale pushes from recreating the old names
gh api -X POST repos/jobovy/galpy/rulesets --input - <<'EOF'
{"name": "feat-backends-retired", "target": "branch", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["refs/heads/feat/backends*"], "exclude": []}},
 "rules": [{"type": "creation"}]}
EOF

# (c) immutable release tags (enforces "no rebase once tagged" on the tag side;
#     the v2 ruleset's non_fast_forward enforces it on the branch side)
gh api -X POST repos/jobovy/galpy/rulesets --input - <<'EOF'
{"name": "release-tags", "target": "tag", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["refs/tags/v*"], "exclude": []}},
 "rules": [{"type": "deletion"}, {"type": "update"}]}
EOF

# (d) disabled "Agent" ruleset: keep it consistent
gh api -X PUT repos/jobovy/galpy/rulesets/19751035 --input - <<'EOF'
{"conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH", "refs/heads/v2"], "exclude": []}}}
EOF
```
Run (a) *before* the rename. That way the `deletion` rule cannot block the rename or the
deletion of `feat/backends-develop`, and v2 is protected as soon as it exists.

### Rename (Jo)
```bash
gh pr close 892 -R jobovy/galpy -c "Superseded: the backend work continues on the v2 branch (galpy 2.0 pre-releases); main stays 1.x."
gh api -X DELETE "repos/jobovy/galpy/git/refs/heads/feat%2Fbackends-develop"   # after (a)
gh api -X POST "repos/jobovy/galpy/branches/feat%2Fbackends/rename" -f new_name=v2
# (or Settings → Branches → rename / the branch list's pencil icon)
```
Local, in every clone and worktree:
```bash
git fetch origin --prune && git branch -m feat/backends v2 && git branch -u origin/v2 v2
```

---

## 2. Versioning

**Today.** The version is a literal string in four files: `setup.py` (`version="1.12.1.dev0"`),
`galpy/__init__.py`, `doc/source/conf.py` and `CITATION.cff`. `.bumpversion.cfg` (bump2version,
see RELEASE_CHECKLIST.md) has:
- parse `X.Y.Z(.{release}{dev})?`;
- a `release` part with values dev/gamma, where `optional_value = gamma` means final;
- the flow `bumpversion release` (1.12.1.dev0 → 1.12.1) and `bumpversion patch|minor` (→ 1.12.2.dev0).

`tools/inject_git_hash.py` adds `+g<hash>` for non-release wheel builds, which is valid for any of the strings below.
pyodide reads `current_version` with awk.

**v2 config** (in `v2-version.diff`): adds a pre-release label part and number ahead of the dev part. The diff:
```diff
 [bumpversion]
-current_version = 1.12.1.dev0
-parse = (?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(\.(?P<release>[a-z]+)(?P<dev>\d+))?
+current_version = 2.0.0b1.dev0
+parse = (?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)((?P<pre_l>b|rc)(?P<pre_n>\d+))?(\.(?P<release>[a-z]+)(?P<dev>\d+))?
 serialize =
+	{major}.{minor}.{patch}{pre_l}{pre_n}.{release}{dev}
+	{major}.{minor}.{patch}{pre_l}{pre_n}
 	{major}.{minor}.{patch}.{release}{dev}
 	{major}.{minor}.{patch}
 
+[bumpversion:part:pre_l]
+optional_value = final
+values =
+	b
+	rc
+	final
+
+[bumpversion:part:pre_n]
+first_value = 1
+
 [bumpversion:part:release]
```
The four version strings go to `2.0.0b1.dev0`.

Dry-run in a scratch repo, starting from 1.12.1.dev0. bump2version 1.0.1 and bump-my-version 1.5.1 gave identical results:

| command | result |
|---|---|
| `major` | 2.0.0b1.dev0 (this is the value the diff sets) |
| `release` | **2.0.0b1** (tag) |
| `pre_n` | 2.0.0b2.dev0 |
| `release` | 2.0.0b2 |
| `pre_l` | 2.0.0rc1.dev0 |
| `release` | 2.0.0rc1 |
| `pre_n` | 2.0.0rc2.dev0 |
| `pre_l` | 2.0.0.dev0 |
| `release` | 2.0.0 |

All four files update. PEP 440 normalizes every string unchanged and they sort in this order. The one exception is
`2.0.0.dev0`, which sorts *before* b1. It only exists as an unpublished step between the last rc and
final; alternatively use `bump-my-version bump --new-version 2.0.0`.

Notes:
- bump2version rewrites the cfg and drops comments, so the cfg has none. bump-my-version keeps the file as it is.
- After 2.0.0 final, `patch` on this config would give 2.0.1b1.dev0. Switch back to the main-style config at that point.

**main keeps 1.x.** main keeps its current `.bumpversion.cfg`, which I re-ran unchanged:
release → 1.12.1, patch → 1.12.2.dev0, minor → 1.13.0.dev0. The v2 config cannot be used on main,
because `patch` would produce 1.12.2b1.dev0. Rule: **never cherry-pick version-bump or release
commits between main and v2.**

**Version check** (`galpy/__init__.py`, in `v2-version.diff`). `latest_pypi_version` drops pre-releases, so a 2.0.0b1
user would never hear about b2. One-line change: keep pre-releases when the running version is one.
For 1.x users nothing changes: 2.0.0bN is filtered out, so they get no "upgrade" nag.

---

## 3. Release mechanics

**Today** (`wheels.yml`):
- cibuildwheel v4.2 builds cp310–cp315 (cp311–cp315 on v2, §6) for manylinux x86_64, macOS x86_64/arm64 and Windows amd64, and smoke-tests each wheel (pip and conda).
- `python -m build --sdist` builds the sdist, which is then built from source.
- `deploy_pypi` runs on `release: created` and uses trusted publishing with no `environment`, so it is branch-agnostic.
- `deploy_s3` publishes dev wheels for `refs/heads/main` only.
- RELEASE_CHECKLIST: tag `v<version>`, push the tag, create a GitHub release, and Actions uploads to PyPI.

**For `v2.0.0b1` no workflow change is needed for PyPI.** A release event on a v2 tag runs v2's
`wheels.yml` at the tagged commit, and PyPI classifies `2.0.0b1` as a pre-release from the version string.
I checked pip's behaviour with dummy wheels 1.12.0 and 2.0.0b1 on a local index:

| command | installs |
|---|---|
| `pip install galpy` | 1.12.0 |
| `pip install "galpy[jax]"` | 1.12.0 (pip warns that 1.12.x has no `jax` extra) |
| `pip install --pre galpy` | 2.0.0b1 |
| `pip install --pre "galpy[jax]"` | 2.0.0b1 |
| `pip install galpy==2.0.0b1` | 2.0.0b1 |

Create the GitHub release with `--prerelease`. That keeps v1.12.x as "Latest" and is the flag the pyodide guard checks.

**Extras.** `galpy[jax]` and `galpy[torch]` already exist on feat/backends but are unpinned. CI is unpinned too and tests the latest
versions (green run 37969048143: jax/jaxlib 0.11.2, diffrax 0.7.2, equinox 0.13.8,
torch 2.14.1+cpu, torchdiffeq 0.2.5, torchode 1.0.1, array-api-compat 1.15.0). The diff sets floors at
the oldest versions anything has been exercised on, which is the dev-box GPU env: jax 0.10.1, torch 2.12.0,
array-api-compat 1.14.0:
```
"jax":   ["jax>=0.10.1", "jaxlib>=0.10.1", "array-api-compat>=1.14", "diffrax>=0.7.2"]
"torch": ["torch>=2.12", "array-api-compat>=1.14", "torchdiffeq>=0.2.5", "torchode>=1.0.1"]
```
The sdist metadata shows these. jax 0.10.1 and diffrax 0.7.2 require Python ≥ 3.11, which is
one reason v2 drops 3.10 (§6).

**Docs install text** (`v2-version.diff`): `pip install --pre "galpy[jax]"`/`[torch]` in both places
in installation.rst, plus `git+…galpy.git@v2`. **main needs a fix too** (`main-docs.diff`): main's
"bleeding-edge" command is `pip install --pre --extra-index-url …/wheelhouse/simple galpy`. Once
2.0.0b1 is on PyPI, `--pre` makes that command install the 2.0 beta, not main's dev wheel. The fix pins `"galpy<2"`
in all three OS tabs; `<2` excludes 2.0.0bN under PEP 440.

**Read the Docs** (Jo, RTD admin). Today the default version is v1.12.0, `latest` is main, and there is no feat-backends version.
- Versions → activate branch **`v2`** (slug `v2`, served at docs.galpy.org/en/v2/). Keep the default on v1.12.x.
- `stable` ignores pre-release tags, so v2.0.0b1 won't take it over.
- If an automation rule auto-activates new tags, `v2.0.0b1` appears as its own version, which is fine.
- No `.readthedocs.yaml` change: notebooks are pre-executed (`nbsphinx_execute = "never"`) and no autodoc page imports jax or torch.
- Do *not* do the checklist's "switch default RTD version" step for b releases.

**conda-forge:** the autotick bot should skip pre-releases. Check that no PR appears on galpy-feedstock.

---

## 4. Checklist (in order)

| # | step | who |
|---|---|---|
| 1 | Merge the in-flight PRs into feat/backends (#1648 #1659 #1661 #1662), or let them be retargeted | me |
| 2 | **Last rebase of feat/backends onto main** (6 commits; reference-tree procedure). This is the last rebase ever. | me prepare, Jo force-push (`feat` ruleset) |
| 3 | Close #892; ruleset (a); delete `feat/backends-develop`; rename; ruleset (b), (c), (d) | Jo |
| 4 | **Freeze merges into v2** until step 5 lands. Without the patched filters, PRs into v2 skip backend-tests but `all-clear` still passes. | both |
| 5 | Apply `v2-workflows.diff` as the first PR into v2. Its own run uses the patched filters (pull_request reads workflows from the merge ref), so it validates itself. | Jo |
| 6 | Push a no-op commit or close/reopen the retargeted open PRs. A base change does not re-trigger CI. | me |
| 7 | PR `v2-version.diff` into v2 (2.0.0b1.dev0, cfg, extras, docs, HISTORY stub, version check); PR `main-docs.diff` into main | me (v2 merge me; main merge Jo) |
| 8 | Optional PR into v2: `.gitattributes` with `HISTORY.txt merge=union` (see risks) | me |
| 8b | Python 3.10 drop (§6): it is already in the step 5 and 7 diffs. After #1648 merges, PR the notebook/paper simplifications from §6 | me |
| 9 | RTD: activate `v2` | Jo |
| 10 | Update my side: 56 memory notes, 8 galpy-jax scripts that reference feat/backends (ledger_sweep.py, g13 burndown measures, assemble_report.py, …), gh-stack bases, STATUS.md | me |
| 11 | First push-to-v2 wheel build is green (wheels.yml on all platforms and cp311–cp315) | me watch |
| 12 | Release PR into v2: HISTORY date and text, `bump-my-version bump release` → 2.0.0b1 | me prepare, Jo text |
| 13 | Tag and release. Tag the squash commit on v2: `git tag v2.0.0b1 <sha> && git push origin v2.0.0b1`, then `gh release create v2.0.0b1 -R jobovy/galpy --verify-tag --prerelease --title "Version 2.0.0b1" --notes-file notes.md` | Jo |
| 14 | Check PyPI: `pip index versions galpy --pre`; in a fresh py3.12 venv, `pip install --pre "galpy[jax]"` and `pip install --pre "galpy[torch]"` plus a smoke test; check that the pyodide `latest` wheels are unchanged | me |
| 15 | PR into v2: `bump-my-version bump pre_n` → 2.0.0b2.dev0 | me |

**Risks**
- **Cherry-pick conflicts.** I dry-ran today's 6 pending main commits onto feat/backends:
  - 3/6 conflict: #1633 (dynamfric + test), #1644 (actionAngleStaeckel), #1651 (diskdf). These are the modules v2 migrated.
  - HISTORY.txt conflicted in all 3.
  - With a `merge=union` attribute on HISTORY.txt, only the code conflicts remain.
  - Expect roughly half of main's code PRs to need a hand-resolved port. That is why step 2 rebases rather than cherry-picks the backlog.
- **Codecov.** The default branch stays main and codecov.yml needs no change. PR comparisons into v2 use the base
  commit's upload, which exists because build.yml now runs on push to v2. Old feat/backends
  history stays under that name in the Codecov UI.
- **Doc notebooks.** `update-doc-notebooks.yml` is hard-wired to main: its preflight uses `--branch main`, and
  schedules only run on the default branch. v2's backend tutorials (#1648) therefore have to be refreshed by hand
  (a local papermill run and a PR into v2) until the workflow gets a `ref`/`base` input.
- **Schedules never run on v2.** The weekly `backend-tests` and `build` crons only fire on the default branch.
  This is already true for feat/backends. Use `gh workflow run backend-tests.yml --ref v2`, as the REGEN runs did,
  or add a cron on main that dispatches it.
- **Wheel CI cost.** wheels.yml now runs 24 builds and 20 tests on every code push to v2. If that is too much,
  switch it to `workflow_dispatch` and run it before each tag.
- **Stale pushes.** A stale clone or worktree pushing `feat/backends` recreates it; ruleset (b) blocks this.
- **Tag immutability.** Once v2.0.0b1 exists, any v2 history rewrite orphans the tag. The rulesets
  make both impossible without Jo disabling them on purpose.

---

## 6. v2 drops Python 3.10 (Jo, 2026-10-09)

galpy 2.x requires Python ≥ 3.11; main (1.x) keeps 3.10. `galpy[jax]` cannot install on 3.10 anyway,
because jax ≥ 0.7 and diffrax 0.7.2 require 3.11.

**In the diffs** (re-verified: both apply cleanly on b20733558, all workflow YAML parses, and no `"3.10"`, `cp310` or `>=3.10` is left
in .github, setup.py, README.md or installation.rst):

| where | change | diff |
|---|---|---|
| `setup.py` | `python_requires=">=3.11"`; drop the `Python :: 3.10` classifier (pyproject.toml has no `requires-python`) | version |
| `README.md` | "galpy 2.x supports Python 3.11–3.14 (galpy 1.x also 3.10)"; the CI sentence drops 3.10 | version |
| `doc/source/installation.rst` | "Python versions >=3.10" → ">=3.11" (2 places) | version |
| `.github/copilot-instructions.md` | supported versions 3.11–3.14 (also fixes a stale "3.9 … 3.13" line); not a workflow file | version |
| `HISTORY.txt` | v2.0.0b1 stub: "Python 3.10 is no longer supported" | version |
| `tests/conftest.py`, `galpy/backend/interpolate.py`, `tests/test_backend_interpolate.py` | comments only: "3.10 through 3.14" → "3.11 …"; "scipy 1.15 (Python 3.10)" → "scipy 1.15" | version |
| `build.yml` | remove the two `ubuntu-latest` 3.10 orbit shards (3.11, 3.12 and 3.13 have the same two) | workflows |
| `build_windows.yml` | remove the two `windows-latest` 3.10 orbit shards | workflows |
| `wheels.yml` | remove `["cp310","3.10"]` from the build and test matrices. `CIBW_BUILD` comes from the matrix, so no `CIBW_SKIP` is needed; cibuildwheel would also skip it via `python_requires` | workflows |
| `test-doc-notebooks.yml` | matrix `["3.11","3.12","3.13","3.14"]` | workflows |

**Nothing to change** in:
- `backend-tests.yml` (3.14 only), `update-doc-notebooks.yml` (3.13) and `build_pyodide_wheels.yml` (3.12–3.14).
- `.readthedocs.yaml` (3.11) and codecov.yml.
- `tools/create_galpy_wheels.sh`: a stale local script pinned to galpy 1.8.0; leave it.
- There are no `sys.version_info` checks against 3.10. The only checks are:
  - `>= (3, 14)` in two tests;
  - `>= (3, 8)` in `_load_extension_libs.py` and `< (2, 7)` in setup.py. Both were already dead before this change; optional cleanup.

**Packaging effect.** On 3.10, `pip install --pre galpy` falls back to 1.12.x, because PyPI serves
`Requires-Python: >=3.11` with the 2.0.0bN files. Users on 3.10 never get a v2 build that fails.
I checked this with dummy wheels: `pip download --pre --python-version 3.10` picks 1.12.0, and `3.12` picks 2.0.0b1.

**Not dead (keep):**
- `galpy/backend/interpolate.py` 1-D-y workaround for scipy 1.15's private GCV. scipy 1.15 still
  installs on 3.11–3.13, so only the comment's "(Python 3.10)" goes.
- `_carries_jvp_tangent`'s `jax_use_direct_linearize` gate in `galpy/backend/_jax/orbit_ode.py`. A user can still have
  jax < 0.7 on 3.11 if they install without the extra, or can turn the flag off.
  `tests/test_backend_inbackend_ode.py:849` tests the flag-off path.

**Follow-up simplifications** (the explicit `{'adjoint': 'direct'}` existed only because 3.10 gets jax 0.6,
where galpy cannot auto-select the direct adjoint under `jvp`/`jacfwd`):
- **#1648 `doc/source/tutorials/orbits/backends.ipynb`:**
  - Markdown cell before cell 32: drop "(such as JAX 0.6, the last version for Python 3.10)" and
    "We select it explicitly here, so that the examples run with every version of JAX". Keep the cases that still need
    `direct`: `jacfwd(jit(f))`, and reverse applied twice.
  - Cell 32 (`jax.jacfwd(Rz_direct)`): drop `inbackend_kwargs={"adjoint": "direct"}`. Auto-selected.
  - Cell 34 (`jax.hessian`): **keep** (second derivative; the docstring says pass `direct`).
  - Cell 36 (`vmap(jacfwd(...))` with `nsteps`): drop `adjoint`, keep `nsteps`. Verify that the tangent probe sees through `vmap` first.
- **#1648 `streams/backends_streamspraydf.ipynb`:** cell 1 `spray()` and the intro text ("so that the examples run with
  every version of JAX") lose `integrate_kwargs={"adjoint": "direct"}` where only `jvp`/`jacfwd` is used.
- **#1648 `streams/backends_streamdf.ipynb`:** cells 5 and 25 (`# for forward mode, see below`). Verify the forward-mode
  path goes through the probe. It is not jitted, so it should; if so, drop the kwarg.
- **Paper (galpy-jax):** drop the "stays for jax 0.6 (Python 3.10)" rationale and the kwarg where only first-order forward mode is used:
  - `paper/ms/examples/streams_02_spray_jvp.py` and `streams_03_spray_track_jacfwd.py`;
  - the streams_02 note in `make_snippets_md.py` (regenerate `SNIPPETS.md`);
  - `paper/ms/sections/streams.tex` :113/:127.
  - Keep it in `gS1_stream_track_sensitivity/make.py` (an outer d/dq through an inner `jacrev`: needs `DirectAdjoint`) and in
    `gS3_track_extrapolation/make.py` (`_IKW`, which sets `max_steps`; check whether `direct` is still needed there).
- Each removal needs a run on jax ≥ 0.10.1 showing auto-selection gives the same derivative (`grad-vs-FD`
  or match the `direct` result), not just that it runs.

## 7. Future: automatic main → v2 port PRs (sketch, not now)

Off-the-shelf option: `korthout/backport-action`. It is label- or always-driven, cherry-picks the merged PR's
commits onto a branch off the target, opens a PR, and on conflict
(`experimental: '{"conflict_resolution": "draft_commit_conflicts"}'`) opens a draft PR
with the conflict markers committed. Hand-rolled equivalent, placed on **main** (the default branch), so Jo applies it:

```yaml
name: Port main to v2
on:
  push:
    branches: [main]
permissions: {contents: read}
jobs:
  port:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/create-github-app-token@v2   # NOT GITHUB_TOKEN: PRs it opens don't trigger CI
        id: app
        with: {app-id: "${{ vars.PORT_APP_ID }}", private-key: "${{ secrets.PORT_APP_KEY }}"}
      - uses: actions/checkout@v7
        with: {fetch-depth: 0, token: "${{ steps.app.outputs.token }}"}
      - env:
          GH_TOKEN: ${{ steps.app.outputs.token }}
          BEFORE: ${{ github.event.before }}
          AFTER: ${{ github.sha }}
        run: |
          git config user.name "galpy-port-bot"; git config user.email "port-bot@users.noreply.github.com"
          pr=$(gh api repos/$GITHUB_REPOSITORY/commits/$AFTER/pulls --jq '.[0].number')
          labels=$(gh pr view "$pr" --json labels --jq '.labels[].name')
          grep -qx 'no-v2-port' <<<"$labels" && exit 0          # opt-out (release/version-bump PRs)
          br=port/main-pr-$pr
          git switch -c "$br" origin/v2
          echo 'HISTORY.txt merge=union' >> .git/info/attributes
          if git cherry-pick -x "$BEFORE..$AFTER"; then draft=; label=port
          else git add -A && git -c core.editor=true cherry-pick --continue || git commit -am "conflicts"; draft=--draft; label=port-conflict
          fi
          git push origin "$br"
          gh pr create --base v2 --head "$br" $draft --label "$label" \
            --title "[v2 port] $(gh pr view $pr --json title --jq .title) (#$pr)" \
            --body "Cherry-pick of #$pr ($BEFORE..$AFTER) onto v2.${draft:+ Conflicts committed with markers: resolve, then mark ready.}"
```

Notes:
- `push` (not `pull_request_target`) runs only trusted code. `BEFORE..AFTER` covers both squash merges (one commit) and rebase merges (N commits).
- The cherry-pick loop needs a fix-up for multi-commit conflicts: resolve per commit, or squash them into one commit with markers.
- The port branch needs to match `backend/**` (or rely on base `v2`) so backend-tests runs.
- Skip `[pre-commit.ci]` autoupdates if v2 runs its own.
