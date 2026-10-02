# Attribution and third-party boundaries

Fly Effect is independent of the projects and institutions below.

The portable LIF adapter follows published equations and parameter conventions from Shiu et al., Nature 2024, [DOI](https://doi.org/10.1038/s41586-024-07763-9). The reference by Philip Shiu and Nico Spiller is MIT-licensed; its notice is retained in licenses/Shiu-reference-MIT.txt. The Fly Effect adapter does not import or redistribute Eon's GPL backend.

Original research code from the earlier Digital Fly Lab / digital-fly-organism work is included. Its notice is retained in licenses/Digital-Fly-Lab-MIT.txt. Neither notice licenses unrelated data or assets.

Optional dependencies retain their own licenses. FlyGym/NeuroMechFly and FlyMimic provide important models and separately acquired resources. No external meshes, muscle XML, trained weights, connectome tables, or upstream code trees are bundled. Runtime-generated files can contain local resource paths and are excluded from releases.

Recorded-run packages that include FlyGym 2.1.0 / NeuroMechFly v2 geometry must retain
`Copyright 2023-2026 The NeuroMechFly v2 Authors`, include the Apache License 2.0 text, identify
rendering or other changes, and avoid implying upstream endorsement. A checked copy of the license
is retained in `examples/hae-recorded-replay/LICENSES/Apache-2.0.txt`.

Recorded values derived from the FlyEM Male CNS connectome (`male-cns:v1.0`) are kept under the
source's CC BY 4.0 attribution conditions. A distributed derivative must credit the HHMI Janelia
FlyEM, University of Cambridge Drosophila Connectomics Group, MRC Laboratory of Molecular Biology,
and Google Research collaboration; link the MaleCNS source and CC BY 4.0; identify transformations;
and avoid describing simulated, reduced, uncalibrated output as raw biological observation. MANC
motor-annotation values derived from Cheong et al. eLife 13:RP96084 carry the corresponding eLife
CC BY provenance. Canonical CC BY links are retained in
`examples/hae-recorded-replay/LICENSES/CC-BY-4.0.url.txt`.

The public replay allowlist excludes the local `observation.npz` source recording, an absolute-path
`body.xml`, raw connectome/MANC tables, raw model meshes, and FFmpeg/libx264 binaries. Compliance
with these terms does not itself authorize publication.

References:

- [NeuroMechFly](https://neuromechfly.org/)
- [FlyGym 2.1.0](https://github.com/NeLy-EPFL/flygym/releases/tag/v2.1.0)
- [FlyEM Male CNS download and license](https://male-cns.janelia.org/download/)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- [FlyMimic](https://arxiv.org/abs/2509.06426)
- [CPG research](https://doi.org/10.1101/2025.09.12.675944)
- [Motor-unit physiology](https://elifesciences.org/articles/56754)
- [Eon's integration and limitations](https://eon.systems/updates/embodied-brain-emulation)

Citations are not endorsements or license grants. Additional material requires a source, revision, and applicable terms before redistribution.
