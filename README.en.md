# Fly Effect

**Small neural connections. Shared scientific understanding.**

[日本語](README.md) · [Getting started](docs/GETTING_STARTED.md) · [Contributing](CONTRIBUTING.md)

Fly Effect is an open research and engineering project exploring interactions between neural circuits, bodies, and environments under limited computational resources. We start with the fruit fly and aim to make experiments reproducible, inspectable, and useful to others.

We welcome contributions motivated by curiosity, education, basic research, simulation methods, and potential future benefits to medical research. We do not claim a complete digital organism, consciousness, or clinical validity.

**Pre-alpha:** a synthetic neural demo, CPU LIF backend, checkpoint utilities, and experimental neuromechanical integration are available. Autonomous walking, learning, and flight are not complete. Biological data, meshes, and trained policies are not bundled. Historical local results do not certify the public package.

    python -m venv .venv
    # Activate the environment, then:
    python -m pip install -e ".[dev,neural]"
    fly-effect doctor
    fly-effect demo --out work/my-first-circuit
    python -m pytest

The three-neuron demo tests software operation and checkpoint replay, not biological performance. Full experiments need separately prepared data: see [setup](docs/GETTING_STARTED.md), [data](docs/DATA.md), and [status](docs/STATUS.md).

Our next milestones are a published CPG reference reproduction, propulsion feasibility on the exact muscle/body model, and evidence-driven coupling of the two. Diagnostic controllers and reduced circuits do not count as final connectome-driven behavior.

Contribute through reproducible bug reports, documentation, small experiments, unit/coordinate checks, visualization, or translation. See [CONTRIBUTING](CONTRIBUTING.md), [GOVERNANCE](GOVERNANCE.md), and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md).

Original code is proposed under MIT. External material retains its own terms. No Eon GPL backend or biological dataset is redistributed here. See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).


Issues and PRs in English or Japanese are welcome. Most detailed documentation currently starts in Japanese; translations are welcome.
