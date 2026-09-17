# Intervention audit: results and interpretation

Date: 2026-09-17. This is a retrospective reanalysis of existing GitHub Actions
run **35189009138**, source commit **787b67537034646edb1f61ffac64da188ecbb17d**.
No new full-CNS simulation or biological experiment was performed for this audit.

## Verified evidence

- The existing experiment retained 166,700 annotated neurons, 25,582,938 connection
  rows and 124,177,617 synapses in its selected population, with 815 motor-model
  outputs. This is not a statement that every raw fragment was included.
- All 90 historical condition records were audited. All 30 archived motor trace
  records at gain 0.8 were recomputed against their error norms and model bounds.
- The rank-4 basis was checked against the SVD of the archived training traces.
- Local tests: **262 passed**, comprising 195 existing tests and 67 new tests.
  Software tests are not biological replications.

The reported metric is **R = 1 - ||candidate-reference||_F / ||lesion-reference||_F**.
It measures error reduction relative to lesion in this dimensionless model. It is
not a percentage of recovered brain function, a biological accuracy score, or a
clinical efficacy measure. Below, ranges span the three gains and three input
seeds per cohort; these are not independent animals.

| Condition | Latent dimension / 64 | R range |
|---|---:|---:|
| Two descending-neuron inputs, POD | 4 | 98.4970% to 99.6573% |
| Independent region-port stress, frozen POD | 4 | -0.0634% to 1.9920% |
| Stress-selected POD | 64 | Numerically approximately 100%; NOT compression |
| Internal degree-preserving shuffle, DN inputs | 64 | 96.0928% to 97.5174% |

The historical stress comparison changes input location and amplitude together.
It cannot isolate their separate causal contributions. The prospective design
separates those factors and has not been executed.

## Aggregate performance hides weak output channels

At gain 0.8, consider motors whose lesion-effect norm is at least 1% of the maximum
motor lesion effect in that trial (a post-hoc diagnostic, not a preregistered gate).
For DN seeds 101, 102 and 103 respectively, **15/537, 19/527 and 7/554** eligible
motor channels had R below 0.95 despite the high aggregate value. Their minimum
R values were **83.2405%, 79.8781% and 90.2382%**. The audit reports ALL three
sensitivity floors, 0.1%, 1% and 10%, rather than choosing a favorable floor.
These are model-output channels, not measurements from living motor neurons.

## New geometric diagnostic

For the archived fixed 64-by-4 linear decoder, an explicitly constructed bounded
input at zero state produces a native first-step region response orthogonal to
the decoder's entire output subspace. Minimum relative region L2 error is **1.0**.
Tests execute the existing ArrayEndpoint on constructed inputs as well as checking
its geometry. The input is designed after the basis is known, not a blind test.

An independent 8,192-input numerical probe lost **93.7713%** of first-step region
response energy, compared with the isotropic linear-algebra value **93.75%**.
This is not a motor-output or biological loss estimate. The argument does not
cover nonlinear decoders or full-rank direct feedthrough and is not claimed as a
new linear-algebra theorem.

## Claim boundaries

High DN performance does not establish universal intervention equivalence.
Rank 64 does not establish compression. A successful shuffled control prevents
this assay from demonstrating unique necessity of the original internal topology.
The imposed contractive rate model is not the upstream Brian2 LIF model and cannot
establish autonomous sustained rhythms or walking. No living tissue is connected.

## Reproduction and provenance

Use the exact historical ZIP (artifact 10483482922), verify its SHA-256, and extract
result.json and traces-gain-0.8.npz. Then run:

```sh
python -m pytest tests/circuit_bridge tests/circuit_bridge_intervention -q
python -m research.circuit_bridge.intervention_audit \
  --result historical/result.json --traces historical/traces-gain-0.8.npz \
  --out work/intervention-audit.json
```

- ZIP SHA-256: `206577c07a08589dddfe860cc289d89477fe042c7a431d4f9fce4d3f9644ec0a`
- result.json SHA-256: `c24b5b659704009787a7d0bbfe435174ba80a6b81a05ae5f7345389399c76d3c`
- NPZ SHA-256: `21821146f22259dc46fe1ca7dba60822069373ab5d8420781462561ccb6b494d`

The historical artifact is listed to expire on 2026-10-01. The delivery bundle
preserves its exact bytes; a durable DOI archive remains pending. The workflow
fails rather than silently replacing missing evidence with a different run.
