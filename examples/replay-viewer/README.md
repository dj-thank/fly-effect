# Recorded-run replay viewer

This is a data-only viewer for a previously recorded run. It does not run a
simulation, interpolate poses, or claim that the recording passed a behavioral
acceptance gate.

Provide these local files beside `index.html`:

- `replay-data.js`: assigns one validated object to `window.REPLAY_DATA`.
- `motion.mp4`: a video whose frame selection and hash match the data receipt.
- `poster.png`: optional poster rendered from the same recording.

The required data fields are documented in `replay-data.example.js`. Replace
the example only with a generated artifact that records source hashes, frame
selection, units, acceptance status, and third-party provenance. Do not bundle
connectome tables, model meshes, recordings, or rendered derivatives unless
their separate redistribution terms have been checked and their notices are
included.

For a mixed-license recorded-run package, keep these files reachable from the
viewer and list them in a hash-and-size manifest:

- `NOTICE.txt`: upstream copyright, attribution, license, modification, and
  simulated/uncalibrated disclosures.
- `SOURCE_ATTRIBUTIONS.json`: machine-readable sources, versions, URLs,
  licenses, transformations, and affected files.
- `LICENSES/Apache-2.0.txt`: the full Apache 2.0 text when FlyGym 2.1.0 /
  NeuroMechFly v2 rendered geometry is included.
- `LICENSES/CC-BY-4.0.url.txt`: canonical CC BY 4.0 deed/legal-code links for
  MaleCNS/MANC-derived values, together with the required source citations.

The package must state that changes were made and that the output is a
simulated, uncalibrated recording rather than raw biological observation.
Keep `observation.npz`, absolute-path `body.xml`, raw connectome/MANC tables,
and FlyGym meshes out of the viewer package. Repository copies of the relevant
license references are in [`licenses/`](../../licenses/).

Open `index.html` directly. The example intentionally has no real media or
biological dataset.
