# FE-02 bounded independent foot placement

## Decision and boundaries

PR #17 established zero static-feasible poses in the preserved common-offset
nine-pose A/B grid, with foot-only contacts confined to hind legs. This is a NEW
protocol, not a reinterpretation or retuning of that grid. The exact 90-tendon
body is reconstructed from audited sources and the root-only unanchored variant
is derived with the existing 468-array model audit. No other compiled parameters,
force bounds, source assets, neural data or acceptance criteria change.

The hypothesis is that independent limited-hinge initial placements can recover
front/middle/hind contact and necessary free-root wrench balance. The neutral pose
is retained as a control. Initial positioning is an explicit diagnostic operation;
no physical motion, dynamic hold, brain execution or walking is claimed.

## Frozen protocol

The versioned record is `examples/fe02-foot-placement.json`; execution verifies it
against the module constants and hashes it before any data collection.

- Ten candidates: the unchanged neutral pose plus nine combinations of XY scale
  0.8/1.0/1.2 and terminal foot-origin height offsets 0/0.15/0.30 in native model units.
- Each candidate starts from the same neutral state. Only existing limited active
  hinges change; the free root and other coordinates stay fixed during IK.
- Analytic MuJoCo body Jacobians and cached kinematics avoid finite-difference
  forward dynamics. At most 80 least-squares function evaluations per target,
  regularization 0.001, solver tolerances 1e-8, joint-limit inset 1e-4 radians.
- Ground placement brackets first contact and tests the SAME five depths as the
  old study: 0.001/0.005/0.01/0.02/0.04 native units. No model margin change.
- Test contact-only root balance using the original friction diamond and
  compression-only ground forces. LP limit 2 seconds; residual threshold 1e-6.
- Select one depth per candidate: root-feasible first, then most distinct feet,
  then shallowest. Only a root-feasible selection enters the ORIGINAL full-DOF
  muscle/contact support solver with unchanged force-length tension bounds.

The worker has a 240-second wall limit; the complete pipeline including source
registration/study and evidence replay has 600 seconds. It reuses the tested Linux
process-group supervisor, 8192 MiB sampled aggregate RSS and per-process address
space cap. Polling is not a guarantee against instantaneous memory overshoot.
Dependency installation is outside that experiment budget; CI has a 15-minute cap.
All failures retain logs and classified receipts. A numerical/resource failure is
not physical infeasibility; an optimizer stopping at its fixed evaluation budget
can still produce a valid but imperfect initial placement, whose error is saved.

## Reproduction

From a checkout with the pinned body and mechanics dependencies installed:

```sh
python scripts/check_experiment.py examples/fe02-foot-placement.json
python scripts/run_foot_placement_study.py --workspace work/foot-placement-new
python -m organism_core.foot_placement --workspace work/foot-placement-new --verify
```

The workspace must not already exist. No source/receipt/checkpoint is overwritten.
CI stores protocol, code/protected-input hashes, source observations, model receipt,
all candidate coordinates, all sampled LP inputs, optimizer diagnostics, force
accounting, and replay results. Raw meshes/XML/biological data are excluded.

## Interpreting outcomes

`no_root_balance_in_fixed_candidates` rejects only these placements under the
stated contact model. `root_balance_found_muscle_support_unresolved` establishes
the necessary root condition but not muscle support. `static_pose_found` means
one saved full-DOF static force solution, NOT dynamic standing or walking.
`inconclusive`, `blocked` and `invalid_experiment` remain distinct. Never count
unit-test success, IK convergence or a rendered pose as biological progress.

Saved-input replay runs without MuJoCo and checks actual records, input/receipt
hashes, target grid, unchanged coordinates, joint limits, contact labels, selection,
force limits, root and full support. It uses the SAME SciPy/HiGHS family; it is not
an independent physics engine, source authenticity proof or biological validation.

## Integration

This addition is based on PR #17 and does not absorb the independent main-based
runtime-quality PR #18. Keep those review decisions separate. Review the dependency
chain #14 -> #16 -> #17 before integration; #15 is a separate process-document PR.
No merge or default-model switch is performed by the experiment.

## Observed result and evidence follow-up (2026-09-11 JST)

The initial implementation commit `d9018af23532d1659af144d17b4d10749b46173a`
completed all three workflows; the placement run is 34498565638. Artifact
10160866514 has SHA256
`469979c3f2516456a114ec611208846369ee5f98a624a991fbcda95e3102ce45`.

All ten candidates and fifty depth samples completed. All nine independently
placed candidates reached six foot-only contacts and a root-feasible selected
depth; sixteen of fifty depth samples were root-feasible. The neutral control
remained root-infeasible. **Full 72-DOF support was infeasible in all nine selected
candidates under the unchanged 90-tendon bounds.** Six contacting feet do not
mean six nonzero force-bearing feet, an equilibrated pose, standing or walking.
The artificial placements at fixed tested penetration depths are diagnostic only.

An independent saved-array calculation reconstructed all fifty root contact maps
from world contact positions and the free-root quaternion. Maximum difference
from the recorded MuJoCo columns was 4.44e-16. Sixteen retained root force witnesses
and nine force budgets were checked; a separately assembled full-support LP using
HiGHS dual simplex found the same nine infeasible outcomes. This is the same
SciPy/HiGHS family, not independent physics or biological evidence. The archive
has 648 arrays and SHA256
`ed66a3ace24441e5055c9cf454b1fbb093f4f84de438f3c8b98a0fe1cfaf9fcd`.

Review also found six malformed-copy cases accepted by the initial replay:
unsupported walking claim, failed model invariants, failed force audit, a zeroed
force witness, contradictory support status, and altered units. The follow-up
checks the stored witness (not just a new LP), independently reconstructs contact
moments, reuses the established full-support and force-budget evidence checks,
and validates the preregistered protocol, provenance, status and capability flags.
All six copies are now rejected and failed replay replaces any stale success
record. Original inputs, models, protocol and solver tolerances are unchanged.

The follow-up adds 52 synthetic regression tests, including a complete 10x5
negative fixture, rotated-root moment controls, force/friction violations and
stale-success cleanup. Local validation: 261 passed, nine backend-dependent
modules skipped. No local MuJoCo/FlyGym/Brian2 execution is claimed; the exact
new commit must be checked in all existing CI jobs and the placement job.

The next discriminating experiment is passive distal-joint/load-transfer
equilibrium at these fixed contacts and unchanged model constants. Necessary root
balance has been recovered; muscle bounds must not simply be raised to conceal
the remaining joint equilibrium problem. This next experiment is not run here.
