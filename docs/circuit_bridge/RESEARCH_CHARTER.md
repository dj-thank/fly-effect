# Circuit Bridge: intervention-aware functional replacement

Status: research programme and retrospective audit, 2026-09-17. Not a submitted
paper, preregistered experiment, independently replicated finding, or biological
validation. This research branch is separate from the original organism work.

## The scientific question

When does replacing a neural subcircuit preserve its responses to interventions,
rather than merely reproducing outputs on a narrow stimulus distribution?

Our target contribution is an intervention-aware, output-specific criterion for
functional replacement, tested against held-out perturbations in connectome-
constrained systems. It is NOT another claim that a connectome can be loaded,
that a small model can oscillate, or that matching a frequency proves equivalence.
A human-derived tissue interface is a future experimental application, not an
established property of this software.

## Novelty boundary and related work

| Prior work | Already established or investigated | What this programme must add |
|---|---|---|
| Lappalainen et al., Nature (2024), DOI 10.1038/s41586-024-07939-3 | Connectome-constrained mechanistic networks predict fly visual responses | Intervention-preserving replacement with independently held-out outputs |
| Beiran and Litwin-Kumar, Nature Neuroscience (2025), DOI 10.1038/s41593-025-02080-4 | Biophysical uncertainty and degenerate dynamics in connectome-constrained networks | Explicit tests across model assumptions, not same-equation recovery alone |
| Pospisil et al., Nature (2024), DOI 10.1038/s41586-024-07982-0 | Connectome-informed inference of causal effective connectivity | A replacement assay; do not relabel the effectome concept as our invention |
| Pugliese et al., bioRxiv, DOI 10.1101/2025.09.12.675944, v2 linked by author repository | Connectome simulations identify a candidate fly walking CPG | Reproduce and credit it, rather than claiming a new CPG discovery |
| NeuroMechFly v2, Nature Methods (2024), DOI 10.1038/s41592-024-02497-y | Embodied sensorimotor simulation | Calibrated brain-to-muscle coupling and external behavioral validation |

The bounded nullspace witness below is elementary linear algebra. It is a
useful falsification diagnostic, not a new general impossibility theorem or a
claim to priority over model reduction, observability, or causal inference.
A first-in-the-world assertion would require a substantially broader priority
review and independent expert assessment. None is made here.

## What we can currently claim

We reanalysed GitHub Actions run 35189009138, source commit
787b67537034646edb1f61ffac64da188ecbb17d. The full archived experiment had already
completed; this change does not rerun it or take credit for generating it.
Its explicit contractive dimensionless rate hypothesis retained 166,700 annotated
neurons, 25,582,938 connection rows and 124,177,617 synapses in that selected
population. Fragment endpoints outside the annotated population were excluded.
Original source ordering and four upstream byte-lock checks were retained.

The historical model is NOT Brian2 LIF or a physiological reconstruction. Its
transmitter sign assumptions, zero functional weights for unmodelled transmitters,
and imposed contractivity matter. It cannot establish autonomous fly rhythms,
walking, or biological compression.

Across three gains, rank-4 POD performed well on two descending-neuron inputs
but failed on independent region-port stress. A stress-selected rank of 64 out
of 64 is NOT compression. A directed degree-preserving shuffle also passed the
old 0.95 aggregate criterion. This assay therefore does not establish that the
exact internal topology is uniquely required. POD retains the internal operator
and reconstructs all outputs; no compute speedup or black-box identification
has been demonstrated.

The new audit checks all 90 historical condition records, recomputes 30 motor
trace records at gain 0.8, verifies a basis against training-only SVD, reports
per-motor errors at all three declared effect floors, and rejects corrupted or
incomplete evidence. Seeds are input realizations, not animals.

## A constructive failure diagnostic

Assume a zero initial region state, n independently driven region ports, a
fixed linear decoder U with k<n orthonormal columns, and observation of all n
region outputs. The native first step is y = alpha*tanh(u).

Choose a unit vector v perpendicular to range(U), a permitted amplitude A>0,
c = tanh(A)/max(abs(v)), and u = atanh(c*v). Then max(abs(u))<=A and

    y = alpha*c*v,     U.T@y = 0,
    min_z ||y-Uz||_2 / ||y||_2 = 1.

This bounded input is a counterexample to universal all-region-output equivalence
for this fixed rank-deficient decoder. The existing ArrayEndpoint is independently
executed in tests. For independent identically distributed symmetric drive,
E[||(I-UU.T)y||^2]/E[||y||^2] = 1-k/n. The 8,192-vector diagnostic agrees with
that identity for the archived 64-by-4 basis.

This does not prove motor-output error, failure under natural sensation, a bound
for nonlinear decoders, a bound for full-rank direct feedthrough, or biological
non-replaceability. A compressed state with suitable direct input/output paths
might evade this particular obstruction. Testing that escape route is useful.

## The decisive next study: avoid confounded success

The accompanying prospective_protocol.json is a design candidate; it is NOT
an external preregistration and has not been executed. Historical outcomes are
already known. Before confirmatory evaluation, freeze model selection, exact
intervention IDs, primary readouts, effect floors and stimulus manifests. Keep
an external hidden challenge separate from publicly committed seeds.

1. Factor intervention location/count independently of amplitude. Include both
   matched per-port amplitude and fixed total input energy. The old DN-versus-
   region-stress comparison changes multiple factors at once.
2. Compare full rank, native reconnection, lesion, degree-preserving shuffle,
   random subspaces, POD, and an identified dynamical alternative with matched
   parameter/training budgets. Rank n is a parity control, never compression.
3. Measure paired intervention responses, not frequency alone. For each stimulus
   s and perturbation p, Delta(s,p)=Y(s,p)-Y(s,none). Compare both unperturbed Y
   and Delta, using whole-trace and predeclared per-output errors. No post-hoc
   phase shifts, favorable channel omission, or denominator clipping.
4. Test mechanistic uncertainty. Keep anatomical IDs and boundary edges fixed,
   but evaluate calibrated spiking and rate-model hypotheses separately. Do not
   carry a contraction certificate over to an oscillatory or spiking model.
5. Distinguish internal-state equivalence, boundary-output equivalence, motor
   equivalence and behavioral equivalence. Theories should incorporate which
   internal errors can actually reach a predeclared output, not assume that any
   discarded direction necessarily harms behavior.

## Success, failure, and statistics

The proposed 0.95 lesion-normalized criterion is a modelling tolerance, not a
biological or clinical threshold. Confirmatory thresholds and measurement noise
floors must be fixed before untouched data are seen. Near-zero lesion effects
are uninformative, not successful rescue. Report negative values unchanged.

Report every planned condition, failures and excluded conditions with reasons.
Report uncertainty over independent animals only when such data exist; clustered
or paired analyses must reflect the actual sampling hierarchy. More random seeds
in one connectome do not produce independent biological specimens.

A replacement method is interesting only if it predicts its success/failure
boundary and improves over matched controls under unseen interventions, without
restoring full rank or evaluating the full native dynamics while claiming a
speedup. A negative result remains publishable evidence only to the extent that
its scope and novelty are independently justified.

## Biological and external validation gates

- First: reproduce a complete, checksum-locked reference dataset/model, including
  boundary inputs. The earlier four-cell MANC prefix is not sufficient evidence
  for an isolated biological circuit or cross-specimen replication.
- Next: use physiological observations excluded from parameter calibration.
  Map cell types across specimens with provenance; never equate unrelated IDs.
- Body stage: preserve upstream support, muscle calibration and force/kinematic
  acceptance gates. The neural circuit must generate motor outputs; an external
  gait controller cannot be presented as a connectome-generated behavior.
- The NeuroMechFly v2 paper identifies public walking kinematics in Harvard
  Dataverse, DOI 10.7910/DVN/3MCEYR. Retrieval, licensing review, animal/trial
  splitting and validation against those data are still pending. Do not confuse
  simulated controller outputs in a deposit with independent biological data.
- Living tissue requires an authorized laboratory, calibrated devices, safety
  oversight and appropriate ethics review. No tissue has been connected and no
  partner, device access, approval, or experimental funding has been secured here.

## Publication-quality outputs

Target outputs: a theory with explicit assumptions and counterexamples; a locked
multi-model intervention benchmark; an external physiological or behavioral
prediction test; and independent reproduction from a clean environment. Release
code, source licenses, data manifests, all condition tables and negative results.
A code/data DOI, preprint submission, external preregistration and independent
replication are not yet complete. Journal acceptance and global impact cannot
be inferred from test counts or the present audit.

The current GitHub experiment artifact expires 2026-10-01. Its exact archive and
SHA-256 are preserved in the delivery bundle; a durable third-party archive is
still a release gate. Do not silently substitute a regenerated archive.
