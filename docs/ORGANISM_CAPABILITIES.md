# Toward life and consciousness

[日本語](ORGANISM_CAPABILITIES.ja.md)

**Fly Effect's long-term ambition includes realizing life and consciousness in a computational substrate.** We will investigate whether, and under what conditions, that ambition is achievable. Memory, homeostasis and self-maintenance are intermediate milestones, not substitutes for this ambition.

This is a research objective, not a claim about the current software. Completing F01–F16, passing a behavioral benchmark, or saving a checkpoint would not by itself establish life or consciousness.

## Three distinct questions

| Question | What can be tested | Interpretation |
|---|---|---|
| Functional capabilities | Persistent learning, state regulation, anticipatory avoidance, causal pathway controls | Specific model behavior |
| Life | Explicit definitions and models of organization, self-maintenance, adaptation and other proposed criteria | Definition-dependent research evidence |
| Consciousness | Indicators derived from competing theories, with interventions and counterevidence | Theory-dependent evidence, not an automatic certification |

A chemical working definition of life does not automatically apply to a computational system; definition choices must remain visible. Consciousness assessment likewise involves substantial theoretical uncertainty. We will not turn a checklist or a self-report into a definitive consciousness score. [Life-definition context](https://science.nasa.gov/universe/search-for-life/life-on-other-planets-what-is-life-and-what-does-it-need/) · [Consciousness indicators](https://pubmed.ncbi.nlm.nih.gov/41219038/)

## The proposed functional loop

    environment → external sensation → neural circuit → action → body
        ↑                                                  ↓
    future decisions ← learned changes ← modulation ← internal state

The initial state vocabulary is energy, hydration, damage, fatigue and sleep pressure. Units, viable ranges, costs and recovery dynamics must be declared as measured quantities or model assumptions. Operator telemetry may expose these values; a separate action policy must not bypass the neural circuit by directly choosing actions from them.

Resource recovery must follow evidenced ingestion, not merely contact with a food patch. Energy accounting must include declared basal and activity costs, including any isometric cost assumption; negative mechanical work must not silently create energy. Viable ranges are preferable to an unexplained symmetric penalty around an arbitrary point. A homeostatic error may be a teaching-signal hypothesis, not a motor command.

Internal-state modulation has biological motivation, but selecting five variables does not reproduce physiology. A mushroom-body review describes state-dependent modulation involving several input and output pathways; one universal dopamine scalar is an approximation. [Body-state review](https://pubmed.ncbi.nlm.nih.gov/38876486/) · [Peptidergic signaling](https://pubmed.ncbi.nlm.nih.gov/39956367/)

## Learned memory

Begin with one documented mushroom-body compartment and a whitelist of actual KC→MBON edges and DAN targets. Preserve baseline connectivity and weights; store bounded, sparse plastic changes and any eligibility state separately.

Short and long memory components are candidate model choices. Two decaying arrays alone do not reproduce the interactions between distinct memory circuits described in experiments. Establish the simplest local rule before adding consolidation. [Memory-circuit study](https://pubmed.ncbi.nlm.nih.gov/39038490/) · [Cell-type tools](https://elifesciences.org/articles/94168)

Neuromodulated local learning can still belong to the broad family of reinforcement-learning approaches. The architectural requirement is that a separate learned motor policy must not replace the connectome as the behavioral controller. [Homeostatic learning perspective](https://arxiv.org/abs/2507.04998)

## Acceptance protocols

These are proposed test specifications, **not implemented or passing capability tests**.

- **Memory:** paired versus unpaired conditioning, plasticity disabled, memory reset, and washout controls. Test lasting effects after checkpoint restoration. Match test physiology, transient neural state, environment and randomness when isolating learned differences.
- **Homeostasis:** preregister viable ranges, duration, seeds and meaningful effect sizes. Evaluate regulation in held-out environments against interoception-cut and no-resource controls, without an external destination command.
- **Self-maintenance:** compare naive and experienced instances using predictive hazard cues and resource-risk trade-offs. Measure avoidance and viability duration. Do not hardcode the desired avoidance policy or set a self-preservation flag as the result.
- **Individual histories:** train A and B from the same initial graph and parameters with different experiences, then compare them under controlled test conditions. Persisted identity is an experimental bookkeeping concept, not proof of subjective personal identity.
- **Life/consciousness:** maintain a separate theory-indexed evidence dossier, alternative explanations and counterevidence. Functional success does not automatically change this assessment.

No biological genetic state is currently simulated. “Same initial individual” therefore means the same declared initial model and state, not an experimentally verified identical genotype.

Self-maintenance concerns behavior inside the simulated environment. Pause, reset, checkpoint and shutdown remain under operator control. It does not mean resisting operator intervention, uncontrolled replication, or acquiring host resources.

## Implementation sequence

The [machine-readable program](../specs/organism-program.json) contains ten proposed vertical slices and blocking edges. The existing CPG and exact-body positive controls remain foundational. Internal-state accounting and local memory can be developed as bounded component studies while those foundations are established. Embodied claims wait for the corresponding neural and physical prerequisites.

Modules such as metabolism, interoception, neuromodulation, plasticity and ecology are possible implementation seams, not six independent features to add blindly. Each slice must demonstrate a causal end-to-end result at its stated scope.

The current MaleCNS graph already includes brain and nerve cord. FlyWire and BANC findings are useful references, not interchangeable cell IDs or a reason to silently switch datasets. [Brain-and-cord research](https://www.nature.com/articles/s41586-026-10735-w)

The next step is to turn the reviewed slices into issues and implement their measurable tests. No new memory, homeostasis, self-maintenance, life or consciousness capability is certified by this document.

