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

Open `index.html` directly. The example intentionally has no real media or
biological dataset.
