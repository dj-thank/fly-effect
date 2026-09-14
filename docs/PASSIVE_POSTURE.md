# Passive-joint posture equilibrium search (FE-02)

## Question and scope

Can passive-joint posture adjustment **inside the original joint ranges**,
together with contact load sharing, satisfy the root-plus-passive equilibrium
that is infeasible at all nine retained PR #19 placements? This follows the
passive-conflict diagnosis: the fixed poses fail only at 2-3 passive rows plus
the root equations, so the bounded question is whether a nearby declared posture
removes that conflict before any muscle, force-limit or neural change.

The implementation is `organism_core/passive_posture.py`. It reuses the
unanchored-root 90-tendon candidate derived by the unchanged `derive_xml`/
`audit_models` path, `support_diagnostics.contact_columns`,
`static_support.solve_support` and the placement root-balance LP. No mass,
tendon path, tendon limit, spring/damper, friction, solver tolerance, neural or
source value is modified. This is a diagnostic initial-placement search: no
dynamic hold, standing, walking, CNS execution or biological equivalence is
claimed by any outcome.

## Search variables and declared trials

Only DOFs whose tendon-map row is exactly zero may move — the passive tarsus
chain, 30 hinge DOFs in this transferred model. The free root keeps the saved
candidate's XY position and orientation; actuated joints keep saved values.
Per retained root-feasible candidate the declared trial list is fixed before
execution:

1. **Control**: the unchanged saved candidate pose, re-derived first.
2. **Sweep**: every passive DOF, one at a time, at five fixed points over
   `[lower+1e-4, upper-1e-4]` rad, ordered grid-point-major so an early budget
   stop still covers every DOF coarsely.
3. **Seeded**: eight uniform samples over the whole passive box, seed fixed.
4. **Composed**: one posture using each swept DOF's best recorded
   passive-subsystem residual; unscored DOFs keep the saved value. A bounded
   heuristic trial, never an optimum claim.

Each trial re-brackets first ground contact (±2 native, 25 bisection steps),
samples the unchanged five-depth grid, then applies gates in the fixed order:
foot-only contact → root balance → root+all-passive-row subsystem → original
full 72-DOF/90-tendon support. The subsystem and full LPs run on every
root-feasible depth, not only the selected one. Original force bounds, friction
diamond and tolerances are unchanged.

A separate **fixed-geometry spring-authority screen** first replays the saved
support matrices: for each passive row, does any single-row spring shift inside
its achievable interval make the root+passive subsystem feasible while contact
geometry and bias are held fixed? It is labelled non-physical — moving a joint
really moves the contact point and bias — and exists to show whether the
conflict even lies inside passive spring authority.

## Budgets and outcomes

Per candidate: 2400 trial solver calls, 160 screen calls, 120 s wall. Worker:
1200 s overall, 8192 MiB inherited ceiling, 16 saved support witnesses per
candidate (non-feasible samples; positive gate results always save). The call
cap is sized to cover the full declared trial list (160 trials at no more than
15 calls each), so exhaustion is an abnormal stop, not expected coverage.
Timeouts, budget exhaustion, unbracketed trials and unresolved solver states
are recorded separately and never become physical infeasibility claims.
Distinct outcomes: `full_support_feasible_posture_found`,
`passive_feasible_without_full_support`,
`no_passive_equilibrium_in_bounded_search`, plus `inconclusive` on any
unresolved evidence (including unexecuted declared trials) and
`invalid_experiment` on inconsistent evidence.

Protocol note: the `FE-02-passive-posture-v1` pre-registration (issue #4)
paired ~160 declared trials with a 640-call cap, which could never finish the
list — a pilot execution confirmed the truncation and verified `inconclusive`.
`FE-02-passive-posture-v2` only corrects that bound (and wall limits) before
any result was claimed; the search space, gates and solver are unchanged.

## Evidence and verification

`passive-posture/protocol.json` freezes input/code hashes, versions and all
settings before solving. `passive-posture/result.json` retains every trial's
meta, per-depth contacts, root-balance witnesses, gate reports and statuses.
`observations.npz` retains coordinates, contact maps, targets, tendon maps,
limits, friction and force-audit arrays for saved witnesses. The verifier
(`--verify`) replays every saved LP, regenerates the declared trial list and
composed trial, checks gate order and summary counts, and fails closed on any
mismatch — without MuJoCo. Physical input hashes are checked before and after
the search; the placement evidence is never overwritten.

## Local execution, 2026-09-14 JST

Base: PR #24 head `b04fd8f75c14728e5f99a1aaff813b2d552bfdd3`. Environment:
Windows, Python 3.12.10, NumPy 2.2.6, SciPy 1.15.3, MuJoCo 3.9.0, flygym
2.1.0 at the pinned commit; the Linux process-group supervisor is not
available on this platform, so the module ran directly inside a fresh
workspace. The placement stage first reproduced `19eec0ca…` — the same
observations SHA-256 as the retained run — and its verifier passed.

| Result | Observed |
| --- | --- |
| Candidates processed | 10 (9 root-feasible placements + unchanged control) |
| Declared trials executed per candidate | 160/160 (control, 150 sweeps, 8 seeded, 1 composed) |
| Trial solver calls per candidate | 1006–1336 (cap 2400 never reached) |
| Trial best gate | root balance at most: 1003 trials; foot-only only: 435; none: 2 |
| Depth samples reaching root balance | 1705 of 7200; all decided passive subsystems infeasible |
| Passive-subsystem or full-support feasible | 0 samples |
| Spring-authority screen | completed 9/9; zero rows could resolve alone at fixed geometry |
| Solver-undecided points | one seeded-trial root LP (candidate 9, trial 153, depth 0.02) |

Run record outcome: `no_passive_equilibrium_in_bounded_search`. Verified
outcome on this platform: `inconclusive`, because one of the replayed root LPs
is numerically undecided (HiGHS status 4) rather than feasible or declared
infeasible — its downstream gates never ran, so the bounded search is not a
complete negative here. The Linux CI job may decide that LP differently; the
receipts keep the distinction either way. Nothing in this run claims dynamic
holding, standing, walking, or biological equivalence.

## Combined-authority margin (FE-02-passive-margin-v1)

The per-row screen above asks whether any single passive row's achievable
spring shift could resolve the subsystem alone; none could. The margin
diagnostic (`organism_core/passive_margin.py`) asks the sharper question on
every retained support witness — the nine placement poses plus the up-to-16
posture witnesses per candidate: if **all** passive rows shift their spring
torques **simultaneously**, each inside its achievable interval
`k_d*(range_d - q_d)`, does the root+passive subsystem admit a solution at
this fixed geometry?

The LP keeps tensions, contact forces, friction and scaling unchanged and adds
one scalar `t`: the passive row `d` equality `(C r)_d = b_d` becomes
`(C r)_d ∈ b_d + t*I_d`. The optimal `t` is the fraction of the achievable
shift box the balance would need; `t ≤ 1` means combined authority suffices
at this geometry, `t > 1` quantifies the deficit, and solver-infeasible means
the needed residual direction is unreachable by passive springs at any shift.

### Local execution, 2026-09-15 JST

All 153 saved witnesses replayed; every margin LP was decided. Outcome:
`within_combined_authority`.

| Candidate | Witnesses within authority | Minimum t |
| --- | --- | --- |
| 1 | 17/17 | 0.0017 |
| 2 | 17/17 | 0.0037 |
| 3 | 17/17 | 0.073 |
| 4 | 17/17 | 0.059 |
| 5 | 17/17 | 0.201 |
| 6 | 17/17 | 0.392 |
| 7 | 15/17 | 0.660 |
| 8 | 8/17 | 0.701 |
| 9 | 8/17 | 0.762 |

The conflict is therefore not a passive-authority deficit: coordinated shifts
equal to a small fraction of the achievable intervals — 0.17 % at the best
witness — satisfy the fixed-geometry equations. This does **not** prove a
physical posture exists: moving a joint also changes the contact map `C` and
bias `b`, so the margin only aims the next bounded experiment — a physical
posture move along the recorded `required_shift_native` directions — and a
`t ≤ 1` witness says nothing about dynamics, standing, walking or biology.
An infeasible margin LP (needed direction unreachable) would have been the
strong local negative; none occurred.

## Interpretation limits

A `full_support_feasible_posture_found` outcome means only that a diagnostic
initial placement admits a static force witness under declared approximations.
It is not stable standing, not a movement, and would justify a **separate**
bounded dynamic-hold experiment. A bounded negative result covers only this
declared search space — single-DOF sweeps plus eight seeded samples — not all
postures or the organism goal.

## Reproduction

In a fresh workspace first run the existing bounded placement pipeline, then:

```sh
python -m organism_core.passive_posture --workspace work/foot-placement
python -m organism_core.passive_posture --workspace work/foot-placement --verify
python -m organism_core.passive_margin --workspace work/foot-placement
python -m organism_core.passive_margin --workspace work/foot-placement --verify
```

The dedicated `passive-posture` workflow runs the whole chain and uploads
JSON/NPZ evidence and failure receipts, never meshes, XML or biological data.
