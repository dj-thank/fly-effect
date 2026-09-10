# Reviewed runtime + research integration candidate

This branch tests one question: do the independently reviewed runtime and body
research changes work together, with their original checks retained?
It is a **draft integration candidate**, not evidence of a merge into main,
a biological release, or a replacement for reviewing the underlying PRs.

## Exact adopted heads

- Runtime PR #20: `993ae7bb838aa4cf613189e1fe671cd86ef376ba`, including PR #18.
- Research PR #21: `6e013fb30cb385880b620a7c3317c20b800832fa`, including the
  #14 -> #16 -> #17 -> #19 research dependency chain.

The original branches are not rewritten, retargeted, closed or merged. This
candidate records both parent commits. PR #15's separate process documents
are not fully incorporated; preserve its decision-oriented review rules when
integrating those documents rather than restoring its older status snapshot.

## Conflict resolution

Keep the complete research tree, then adopt the eight runtime implementation,
test and documentation files declared in `integration-manifest.json` byte for
byte. The only combined CI edit retains the four OS/Python matrix jobs,
mechanics-demo, body-probe, tendon-study, root A/B and foot-placement workflows.
Add runtime pip caching and main-push/PR triggering without deleting body jobs.
Require artifacts instead of silently ignoring missing evidence. Add public
experiment-record validation and a source-hash receipt to the core matrix.

The status page preserves historical observations and the latest research
limitations, while marking runtime code as present in this candidate only.
No archived historical movement is promoted to verified autonomous walking.

## Reproducible assembly gate

`python scripts/check_integration.py` checks 19 declared source files, including
brain, graph lock, acceptance criteria, original body and support/placement
solvers. It writes `work/validation/integration.json` on success or failure.
Duplicate/unsafe paths, duplicate JSON keys, wrong source refs, missing protected
entries and byte mismatches are rejected. New code/tests still run normally.

This is a reviewable assembly check, **not a tamper-proof signature or a new
biological acceptance rule**. A deliberate future change needs a new source
commit and reviewed manifest update, not disabling a failing check. CI does not
fetch mutable branch heads to guess what was intended. Git parentage remains
separately reviewable on GitHub.

## Software and physical boundaries

Checkpoint metadata has a configurable 16 MiB default decode cap. Whole-state
uncompressed size is optionally bounded; it is unlimited by default for large
CNS compatibility. The NPY allocation preflight is not an OS-level sandbox.
The removed redundant array copy reduces tracked peak allocation in the
recorded synthetic 16 MB restore test, not necessarily whole-process RSS.

The combined local Python 3.13 environment does not provide MuJoCo/FlyGym or
Brian2; those module tests are skipped locally and run in the configured CI
matrix/jobs. Check the actual CI of this candidate, not just its parents.
CI green, completed diagnostics and main integration are distinct facts.

The latest physical evidence still has 0/9 full muscle-support poses, no stable
standing/walking achievement, and no new whole-CNS execution. Passive-load
conflict diagnosis is a replay of fixed constraints, not a changed physical
model. The next physical experiment concerns passive-joint/contact equilibrium
within the original ranges, with a preserved control and explicit identity.
