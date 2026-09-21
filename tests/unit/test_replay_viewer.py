import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "examples" / "replay-viewer"


def test_replay_viewer_bundles_no_recording_or_third_party_binary():
    files = {path.name for path in VIEWER.iterdir() if path.is_file()}
    assert files == {"README.md", "index.html", "viewer.js", "replay-data.example.js"}
    assert not any(path.suffix.lower() in {".mp4", ".png", ".jpg", ".stl", ".npz"} for path in VIEWER.iterdir())


def test_replay_example_keeps_provenance_and_acceptance_explicit():
    source = (VIEWER / "replay-data.example.js").read_text(encoding="utf-8")
    match = re.fullmatch(r"window\.REPLAY_DATA = (\{.*\});\s*", source, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    meta = data["meta"]
    assert meta["acceptance_passed"] is False
    assert meta["pose_interpolation"] is False
    assert meta["provenance"] == {
        "generated_from_recorded_run": True,
        "third_party_assets_in_render": False,
        "redistribution_reviewed": False,
    }
    assert len(meta["feet"]) == 6
    assert all(len(data[key]) == meta["frames"] for key in (
        "time_s", "contacts", "muscle_force", "cum_spikes", "cum_sensory", "root_xyz"
    ))
