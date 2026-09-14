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

## Directed posture shifts (FE-02-posture-shift-v1)

The margin diagnostic is algebraic — it holds the contact map and bias fixed.
The follow-up module `organism_core/posture_shift.py` performs the physical
version: for every margin witness judged within combined authority (`t ≤ 1`),
each passive joint is actually moved to `q'_d = q_d + delta*_d/k_d` and the
unchanged gate chain is re-evaluated at the new geometry, where moving the
joints really does change which geoms touch and what the bias is.

### Local execution, 2026-09-15 JST

All 133 declared directed trials executed; outcome
`no_shift_reached_equilibrium`, verified `evidence_valid: true` with 133
trial records and 22 saved support witnesses replayed.

| Candidate | Directed trials | Best gate | Trials reaching root balance |
| --- | --- | --- | --- |
| 1–2 | 17 | root balance | 3 |
| 3 | 17 | root balance | 1 |
| 4–5 | 17 | root balance | 2 |
| 6 | 17 | foot-only contact | 0 |
| 7 | 15 | foot-only contact | 0 |
| 8–9 | 8 | foot-only contact | 0 |

No shifted posture reached the passive-subsystem gate. The directed shifts
that resolve the fixed-geometry equations largely *degrade* the physical
posture: candidates 6–9 frequently lost even foot-only contact (gate 0),
because the tarsus-chain rotations that produce the needed spring-torque
change also move the foot tips and perturb the contact configuration the
margin screen was holding fixed. The algebraic "within authority" verdict
and the physical outcome differ precisely in the `C`/`b` dependence the
screen cannot see — the spring-shift direction is entangled with contact
geometry, so balance requires a pose that satisfies both at once, which a
single linearized move does not produce.

## Bounded iterative shifts (FE-02-iterative-shift-v1)

Replaying the margin LP on the shifted poses' saved witnesses showed the
required shift *contracting* at some candidates (candidate 1: t 0.0017 →
0.0001 after one move). `organism_core/iterative_shift.py` iterates the
loop a bounded number of times: at each iterate the unchanged gate chain
runs, the minimum-margin root-feasible sample supplies the next passive
displacement `delta*/k`, and the lineage stops on passive/full feasibility,
loss of root feasibility, margin beyond authority (`t > 1`), a joint-limit
violation, or the iteration cap (6 shifts per seed). Two seeds per
candidate — the lowest-`t` within-authority witnesses — give 18 lineages.

### Local execution, 2026-09-15 JST

All 18 lineages executed to a terminal status (568 solver calls); recorded
outcome `no_convergence_in_bounded_iteration`. The verifier replayed every
iterate — re-deriving each lineage's q sequence from the saved driver
`delta*` witnesses and re-solving every support/subsystem/margin LP —
`evidence_valid: true`, 98 saved witnesses checked. Because four recorded
gate solves (two passive-subsystem and two full-support LPs at deep,
non-driving samples) came back solver-unresolved, the fail-closed verified
outcome is `inconclusive` — the unresolved solves are not counted as
infeasible.

| Regime | Candidates | Margin trajectory | Terminal status |
| --- | --- | --- | --- |
| Contraction | 1–2 | t shrinks geometrically (cand 1: 1.7e-3 → 7.2e-12) | `iteration_cap` |
| Divergence / contact loss | 3–9 | t grows or root feasibility disappears | `no_root_feasible_sample` |

The contraction regime is the scientifically interesting negative: at
candidates 1–2 the required shift approaches zero asymptotically yet exact
passive feasibility is never reached within the cap — each linearized move
lands infinitesimally short because it re-perturbs the contact geometry it
was computed under. No iterate reached the passive-subsystem or
full-support gate; no dynamic trial ran.

## Tension-authority margin (FE-02-tension-margin-v1)

The iterated contraction left an open question: is the residual deficit
muscular (declared tendon tension limits too small) or structural?
`organism_core/tension_margin.py` answers it on every saved support
witness — all equality rows stay strict, the friction diamond is
unchanged, and only the tension bounds are relaxed by a single scalar
`s >= 0` (`f_j <= s*fmax_j`). The optimum `s*` is the muscle-force
multiple the pose and contact set would need for static support;
infeasibility at any `s` means contact geometry alone cannot cover the
rows tendons cannot reach (root translation rows 0-2 and the 30 passive
DOFs have exactly zero tendon-map entries).

### Local execution, 2026-09-15 JST

All 273 saved witnesses (9 placements + 144 posture-search + 22
directed-shift + 98 iterative-shift) decided identically:
`unreachable_at_any_scale` — **no finite muscle-force scale can balance
any saved pose**. Verified `evidence_valid: true` with every LP
re-solved from retained arrays; outcome `unbounded_tension_deficit`.

This closes the deficit decomposition: the transferred muscle tension
limits are NOT the bottleneck. The binding constraint is the contact
cone's ability to cover the tension-free rows exactly — a structural
property of pose and contact configuration, addressable only by posture
or contact changes (the passive-joint iteration approaches but never
lands) or by additional contact points, never by stronger muscles.

## Residual-descent posture generation (FE-02-residual-descent-v1)

With muscle force eliminated as the bottleneck, the only remaining lever is
posture/contact generation — and the actuated (tendon-spanned) joints had
never been searched: the posture search and the margin iteration moved only
the 30 passive DOFs. `organism_core/residual_descent.py` runs two declared
phases. Phase 1 is a generator: cyclic coordinate descent over all 66
non-root DOFs on the passive-subsystem minimum scaled balance error, with
the seed's contact map, tendon map and friction held fixed and only the
bias target `qfrc_bias - qfrc_passive` recomputed per probe. Phase 2 is the
certifier: the terminal poses re-derive contacts and run the unchanged
gate chain (contact -> root -> passive subsystem -> full support, audits
and margin LP included).

### Local execution, 2026-09-15 JST

Four contracting seeds (candidates 1-2, margin t <= 1e-3) descended within
the declared budgets; recorded outcome
`descent_floored_without_feasibility`. **Every accepted move was on an
actuated DOF — 49/49 for candidate 1, 3-5 for candidate 2 — the previously
unsearched actuated space carried all productive directions.** Candidate 1
descended 1.4e-2 -> 2.1e-4 and was still improving when the 3000-eval cap
stopped it (not converged); candidate 2 floored at ~5-6e-6 — under its
saved contact map no coordinate move of any DOF reduces the residual
further, so that contact configuration cannot reach exact zero. Physical
certification reached gate 1-2 only (root balance at best): the
fixed-map optimum does not transfer, as expected — contact points move
with the pose. Verified `evidence_valid: true` with every trajectory and
certification LP replayed; the verified view is `inconclusive` because two
seed-initial LPs are numerically undecided (fail-closed, same solver
boundary seen in prior runs).

Interpretation: even with all 66 non-root DOFs free, fixed contact geometry
cannot reach exact zero — the binding constraint is the contact
configuration itself. Next justified direction: a generator that changes
which tarsus points touch (new contacts appearing), e.g. contact-aware
posture search.

Linux CI reproduced candidate 2 bit-identically (floors ~5e-6) and decided
the two seed-initial LPs that were undecided on Windows, so the verified
outcome there is `descent_floored_without_feasibility` directly. Candidate
1's fixed-map floor resolved on Linux to ~4.5e-9 — at solver resolution
but still nonzero, with 41 unresolved probe evals marking how boundary-
close these subsystem LPs sit.

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
python -m organism_core.posture_shift --workspace work/foot-placement
python -m organism_core.posture_shift --workspace work/foot-placement --verify
python -m organism_core.iterative_shift --workspace work/foot-placement
python -m organism_core.iterative_shift --workspace work/foot-placement --verify
python -m organism_core.tension_margin --workspace work/foot-placement
python -m organism_core.tension_margin --workspace work/foot-placement --verify
```

The dedicated `passive-posture` workflow runs the whole chain and uploads
JSON/NPZ evidence and failure receipts, never meshes, XML or biological data.
