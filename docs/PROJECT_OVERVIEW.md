# Project overview

[日本語](PROJECT_OVERVIEW.ja.md) · [README](../README.md) · [Issues](https://github.com/dj-thank/fly-effect/issues) · [Pull requests](https://github.com/dj-thank/fly-effect/pulls)

Fly Effect is an MIT-licensed, pre-alpha research project. Our long-term ambition is to realize life and consciousness computationally, while investigating whether and how this is possible. Memory, homeostasis and learned self-maintenance are intermediate milestones.

## Current capabilities

The public package provides a synthetic neural demo, checkpoint replay, CPU neural simulation and experimental body coupling. Autonomous walking has not passed acceptance. Learned memory, homeostasis, self-maintenance, life and consciousness have not been established. [Validation scope](STATUS.md) separates package checks from historical research measurements.

Windows/Linux CI covers Python 3.11 and 3.12. Its success verifies software checks, not biological capabilities. [Actions](https://github.com/dj-thank/fly-effect/actions)

## Research work

This is the initial plan published on 2026-09-09. **Issue bodies, labels and discussions carry current work status.** Publishing a specification does not complete its research task.

| Issue | Deliverable | Prerequisites |
|---|---|---|
| [FE-01 #3](https://github.com/dj-thank/fly-effect/issues/3) | Published neural rhythm positive control | None |
| [FE-02 #4](https://github.com/dj-thank/fly-effect/issues/4) | Exact muscle-body propulsion positive control | None |
| [FE-03 #5](https://github.com/dj-thank/fly-effect/issues/5) | Physiological accounting and checkpoint replay | None |
| [FE-04 #6](https://github.com/dj-thank/fly-effect/issues/6) | Internal needs coupled to neural inputs | #5 |
| [FE-05 #7](https://github.com/dj-thank/fly-effect/issues/7) | Ingestion actions restore food/water resources | #5, #6 |
| [FE-06 #8](https://github.com/dj-thank/fly-effect/issues/8) | Persistent association through local plasticity | None |
| [FE-07 #9](https://github.com/dj-thank/fly-effect/issues/9) | State-dependent food/water memory retrieval | #3, #4, #6, #7, #8 |
| [FE-08 #10](https://github.com/dj-thank/fly-effect/issues/10) | Homeostasis in held-out environments | #3, #4, #6, #7 |
| [FE-09 #11](https://github.com/dj-thank/fly-effect/issues/11) | Learned self-maintenance and individual histories | #8, #9, #10 |
| [FE-10 #12](https://github.com/dj-thank/fly-effect/issues/12) | Initial life/consciousness evidence protocol | None |

Start with the published circuit and exact-body positive controls (#3, #4). Physiological accounting (#5), bounded plasticity (#8) and the initial evidence protocol (#12) can be prepared independently. Component success does not certify whole-system behavior. [Detailed research program](ORGANISM_CAPABILITIES.md)

## Branches and participation

- `main` is the shared, reviewed version to read and try. It remains pre-alpha.
- Each change uses a short-lived `codex/<topic>` or contributor topic branch and a PR. Required checks must pass before merging.
- Merged topic branches are removed after their commits are confirmed to be in `main`; PRs preserve review and change history.
- Comment on an issue before substantial work and state your scope to avoid duplicate effort. Unassigned issues are not active implementations.
- `ready-for-agent` means the issue has no outstanding prerequisites; `blocked` means prerequisites remain. Neither label guarantees feasibility or biological correctness.

[Get started](GETTING_STARTED.md) · [Contributing](../CONTRIBUTING.md) · [Discussions](https://github.com/dj-thank/fly-effect/discussions)
