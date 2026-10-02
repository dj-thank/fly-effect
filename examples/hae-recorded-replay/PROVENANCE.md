# HAE recorded-run provenance

Status: `PUBLIC_DISTRIBUTION_AUTHORIZED_BY_OWNER_20260922 / REVIEWABLE_PR`

This package replays the saved `f06-seed1-closed-loop-10s-20260917` simulation.
It does not run a new simulation and does not establish autonomous walking,
biological calibration, a biological clone, or Human acceptance.

## Included transformed outputs

- `assets/actual-motion.mp4` is a 300-frame, 30 fps, silent rendering of selected
  poses from the local source recording. No pose interpolation was used.
- `assets/pose-045010.png` is one rendered simulation pose, not a photograph.
- `assets/replay-data.js` is a reduced, frame-synchronised derivative containing
  contacts, aggregate force, cumulative event counts, and root positions. Its
  units and sensory/load assumptions are uncalibrated.

Changes were made: the local simulation recording was frame-selected, reduced,
serialised, and rendered for browser replay. These files are simulated outputs,
not raw biological observations.

## Separate licenses

- Rendered FlyGym 2.1.0 / NeuroMechFly v2 geometry remains under the upstream
  Apache-2.0 conditions documented in `NOTICE.txt` and
  `LICENSES/Apache-2.0.txt`.
- MaleCNS v1.0 and MANC-derived values retain the CC BY 4.0 attribution and
  citation conditions documented in `NOTICE.txt`, `SOURCE_ATTRIBUTIONS.json`,
  and `LICENSES/CC-BY-4.0.url.txt`.
- Original viewer code remains Fly Effect MIT. That license does not absorb or
  relicense the third-party geometry, source data, or rendered derivatives.

## Deliberately excluded

The source `observation.npz`, absolute-path `body.xml`, raw MaleCNS/MANC tables,
raw FlyGym meshes, and FFmpeg/libx264 binaries are not in this package. Their
absence is enforced by `PUBLIC_FILE_MANIFEST.json` and the package verifier.

The package completed local license integration review. The project owner
authorized publication through a reviewable pull request on 2026-09-22. This
authorization does not promote F06, biological calibration, or Human acceptance.
