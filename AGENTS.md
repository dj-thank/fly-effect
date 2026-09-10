# Agent working agreement

Read `docs/STATUS.md`, `docs/ROADMAP.md`, `CONTRIBUTING.md`, and the target issue before editing. Re-read the live branch head and open PRs; a dated status page is not a live lock or proof that a PR is merged.

## Work on the blocker, not the activity count

- Take one bounded hypothesis from the earliest unresolved dependency. Current FE-02 entry point: `examples/fe02-root-compliance-ab.json` and issue #4. A plan is not an implemented runner or a passing experiment.
- One implementation owner per experiment/file set. Claim issue, base SHA, branch, files and next handoff in an issue comment; inspect existing claims before writing. A reviewer can inspect independently. Unrelated read-only research may continue; do not concurrently mutate the same model or workspace. Claims are coordination, not server-enforced locks.
- Use a new workspace and branch per independent change. Keep a reproducible baseline. For dependent work, declare the base PR/SHA and integrate in dependency order; do not force-push another agent's work or silently retarget a PR.
- One PR answers one decision. Close its evidence/review loop before adding a new research question. If two bounded cycles produce no new discriminating evidence, stop that approach, record why, and choose a different minimal test. Do not stop the overall research goal.

## Invariants

Do not change neural IDs, graph edges/weights, `organism_core/brain.py`, `organism_core/graph_lock.json`, or `ACCEPTANCE.json` merely to make a trial pass. Do not overwrite source XML, meshes, biological data, model receipts or checkpoints. A body variant needs explicit provenance and a new identity; no identity rewriting to restore an incompatible checkpoint. Keep legacy controls reproducible. Do not relax force bounds, tolerances, pose grids or budgets after observing a result.

No root-position forcing in locomotion acceptance. Diagnostic drive, a rendered motion, unit-test success and a valid force budget are not autonomous walking or biological equivalence. Keep the full-CNS baseline; use small controls to isolate failures, not as a replacement brain.

Do not add external data, meshes, weights, papers, credentials or private paths to Git. Do not introduce paid runs, new infrastructure or recurring automation as a side effect of a local improvement.

## Handoff

Report: hypothesis and decision; exact code/model/data identity; executed commands and environment; evidence locators/digests; unexecuted checks and remaining blocker; one next action. Separate infrastructure result, scientific result and integration state. `blocked`, `inconclusive`, and `not_run` must not become pass/fail capability claims. Update `docs/STATUS.md` when the evidence changes, not after every edit.

Follow repository review/check requirements. Do not infer merge authorization from a different project or from a green CI badge. Report changes as a PR until integration is actually confirmed.
