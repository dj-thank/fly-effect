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
