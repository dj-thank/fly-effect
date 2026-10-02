# Passive-load conflict diagnosis (FE-02)

## Question and scope

Which passive joint equations conflict with a fixed pose's root balance, under
the original static support solver? This diagnoses retained PR #19 evidence; it
is not a new trajectory, controller, changed pose, or biological validation.

The implementation is `organism_core/support_conflict.py`. It reuses
`static_support.solve_support` without changing force bounds, friction,
equality scaling or tolerances. All tendon columns and exact values are kept,
including tiny nonzero root entries. A non-root row is called passive here only
when every tendon entry in that row is exactly zero. This is a matrix property,
not a claim that an anatomical joint has no active biological mechanism.

## Deterministic, bounded decision

1. Reverify the physical source evidence using the existing placement verifier.
2. Keep root equations in every solve; check root balance and the root-plus-all-
   passive projection separately. Neither certifies all actuated equations.
3. Check each passive row alone alongside the root equations.
4. Delete rows in ascending order while the remaining subsystem is infeasible.
5. Solve the retained conflict again, then freshly solve each one-row deletion.
   Call it inclusion-minimal only if the conflict is infeasible and every final
   deletion is feasible under the unchanged numerical solver.

This yields **one inclusion-minimal subsystem**, not a minimum-cardinality or
unique cause. Deletion order can choose one of multiple conflicting sets. A
regression fixture explicitly has a one-row conflict that gets discarded in
favor of a different two-row inclusion-minimal conflict. Do not interpret the
last retained leg as the only faulty leg or anatomical cause.

The protocol allows 120 seconds overall and 128 calls per pose. Deadlines are
checked before and after calls. An already executing LP retains its original
10-second limit (and may use a second nearest-balance LP); this module does not
interrupt it. CI adds an outer 3-minute step deadline. Timeout, call exhaustion,
unresolved solver results and inconsistent reports are never converted into a
physical infeasibility claim. Completed subproblems survive an interrupted pose
in the failure receipt. The output directory must be new; old diagnostic
results are never overwritten. The ordinary source verification receipt is
refreshed, while physical input JSON/NPZ hashes must remain unchanged.

## Executed retained-data experiment, 2026-09-11 JST

Source: PR #19 head `c89ac5c95f43e52b4d07c3edc11e8eba8e9d60d3`, placement
artifact `10162237196`, ZIP SHA256
`1ebc70d5f37950246bdd147ff9fd93adc94ed8c2c7d8ee6ff9fa44fd7efa9e71`.

Local execution: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, OpenBLAS/OMP threads 1.
The source verifier passed before the diagnosis; 590 unchanged support-solver
calls completed across nine poses. Physical input hashes were unchanged. This
is an independent replay of saved matrices, not a local MuJoCo/FlyGym run or an
independent physics engine. CI reruns the physical pipeline before this stage.

| Result | Observed |
| --- | --- |
| Root balance feasible | 9/9 poses |
| Root plus all 30 passive rows infeasible | 9/9 poses |
| All individual passive rows feasible in isolation | 7/9 poses |
| At least one individually infeasible row | poses 3 and 6 |
| Freshly verified inclusion-minimal conflicts | 9/9 poses |
| Retained conflict size | 2 or 3 passive equations, in addition to root |
| New trajectories, standing, walking, CNS execution | none |

The ascending-order retained row IDs are 69/70/71 for poses 1-5 and 9,
69/70 for poses 6-7, and 69/71 for pose 8. Pose 3 also has individually
infeasible rows 12/23/45/56; pose 6 has individually infeasible rows 34/67.
These alternative conflicts are why the retained final IDs must not be read
as the unique cause. Row IDs are generalized-coordinate indices in these
saved arrays, not neuronal or biological joint IDs.

Friction is the original inner diamond approximation. Dry joint friction and
joint-limit forces are not added optimization variables. These negative results
apply to the fixed saved poses and declared approximation, not to every possible
pose, controller, or living animal.

## Reproduction and evidence

In a fresh workspace first run the existing bounded placement pipeline, then:

```sh
python -m organism_core.support_conflict --workspace work/foot-placement
```

`passive-conflict/protocol.json` captures input/code hashes, budgets and versions
before solving. `passive-conflict/result.json` retains row selections, original
solver reports and their force witnesses, final deletion checks, outcome and
elapsed time, including failures. The placement workflow uploads these JSONs
with existing raw evidence. Never edit a saved force vector to make it pass.

The next physical question is whether passive joint geometry/rest pose and
contact load sharing can reach equilibrium within the original limits. Compare
explicit variants with an unchanged control and new identity; require full
muscle support before a separate dynamic hold. Do not increase muscle strength
or alter neural weights merely to mask a zero-tendon-row conflict.
