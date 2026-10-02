# Zero-velocity force accounting (FE-02)

## Preregistered question and scope

Does the exact 90-tendon body's support target include passive forces on its free
root, in addition to gravity? This follows the unmodified-model contact/pose
study at `61edb399ea9830d6bc5c9d6ad96629bcc4b0ba91` (PR #14, issue #4).
A hypothesis to test is that a compliance default also reaches the root joint.
This is **not yet an experimental finding** in this protocol. Do not zero any
parameter to improve a result before accounting for the unchanged model.

No neural IDs, graph edges, weights, acceptance thresholds, source XML or meshes
are changed. No biological neural dataset is needed. The deterministic audit uses
no RNG or fitted parameters. Native units and the inherited mass interpretation
remain assumptions, not new animal measurements.

## Implementation and controls

`organism_core.force_audit.audit_static_forces(model, qpos)` creates private
`MjData`, uses the supplied position with zero velocity/input, and records:

- Bias/gravity, spring, damper, gravity compensation, fluid and any available
  adhesion term; the sum is checked against total passive force.
- Independent mass-times-gravity projections through each body's COM Jacobian.
- Applied generalized/Cartesian forces, actuation, total constraint reaction and
  inertial force `M*qacc`; inertia is essential because the pose need not be at
  equilibrium. Generalized torque rows keep MuJoCo's DOF basis.
- Mass, inertia, spring-reference pose, joint/tendon passive settings and the
  free-root stiffness, damping, friction and armature. A root-anchor-parameter
  flag describes those joint parameters only, not all possible world couplings.

Free-fall, root-spring, internal-joint-spring, gravity-disable, gravity-compensation,
world-tendon and rotated-body controls are analytic engineering tests. They do
not stand in for the fly model or demonstrate animal behavior.

`evaluate_support` now rejects nonfinite state/forces, nonzero velocity,
external forces, actuator force, control or activation instead of silently
omitting them from `bias - passive`. This does not establish equilibrium.

## Budget, checks and falsification

The existing pose-study wall budget remains 120 seconds (maximum 300); its
subprocess is bounded at 180 seconds. Nine fixed hip/knee initial placements,
five fixed contact depths and all existing force bounds remain unchanged. Audits
are performed for the recorded pose and each selected foot-only pose.

Gravity projection: absolute and relative tolerance `1e-10`. Passive-component
sum: absolute tolerance `1e-10`. Dynamic balance: per-row scale
`max(1, |M*qacc|, |bias|, |passive|, |constraint|)` and residual at most `1e-9`.
Source support target must reproduce at absolute/relative tolerance `1e-12`.
Nonfinite data, solver warnings, changed parameter snapshots or failed checks
fail the experiment. Unexpected anchors are **reported, not silently removed**;
accounting can pass even if the model is not a freely supported body.

Full vectors and passive parameter arrays are saved in `pose-study/observations.npz`
with `recorded_force_` and `pose_XX_force_` prefixes, alongside the unchanged LP
inputs. The JSON receipt includes source/body/code/observation hashes and scoped
force-budget summaries. A zero-velocity forward evaluation is instantaneous,
not a replay of historic force, a time average or a standing certificate.

## Run

From a repository environment with its pinned body/mechanics dependencies:

```bash
python -m pytest -q
python scripts/run_mechanical_study.py --workspace work/force-accounting --duration 0.25
```

Use a new workspace. Existing CI retains `pose-study/` including failure receipts.
Do not force restoration of an old whole-system checkpoint by changing its code
identity. No standing, propulsion, neural closed-loop or biological gate is
passed by these engineering checks.

## Primary references

- [MuJoCo equations of motion](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework).
- [MuJoCo data and passive force fields](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).
