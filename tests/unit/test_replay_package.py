import hashlib
import json
from pathlib import Path

import pytest

from scripts.replay_package import (
    AUTHORIZED_GATE,
    EXPECTED_FILES,
    build_manifest,
    verify_package,
)
from scripts.audit_release import reviewed_binary_paths


ROOT = Path(__file__).resolve().parents[2]


PROVENANCE = {
    "recording_kind": "simulated_recorded_run",
    "calibration_status": "uncalibrated",
    "source_recording_included": False,
    "raw_inputs_included": False,
    "changes_made": True,
}


def replay_data():
    return {
        "meta": {
            "frames": 3,
            "fps": 30,
            "duration_s": 0.1,
            "feet": ["lf", "lm", "lh", "rf", "rm", "rh"],
            "source_sha256": "0" * 64,
            "video_sha256": hashlib.sha256(b"assets/actual-motion.mp4").hexdigest(),
            "pose_interpolation": False,
            "walking_passed": False,
            "force_units": "native (uncalibrated)",
            "total_spikes": 5,
            "total_sensory": 3,
        },
        "time_s": [0.001, 0.034, 0.068],
        "contacts": [[0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
        "muscle_force": [0.0, 0.5, 0.25],
        "cum_spikes": [0, 2, 4],
        "cum_sensory": [0, 1, 2],
        "root_xyz": [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
    }


def write_replay_data(root, data, provenance=PROVENANCE):
    (root / "assets/replay-data.js").write_text(
        "window.REPLAY_PROVENANCE = " + json.dumps(provenance) + ";\n"
        "window.REPLAY_DATA = " + json.dumps(data) + ";\n",
        encoding="utf-8",
    )


def make_package(root: Path) -> None:
    links = " ".join(
        (
            "NOTICE.txt",
            "PROVENANCE.md",
            "SOURCE_ATTRIBUTIONS.json",
            "LICENSES/Apache-2.0.txt",
            "LICENSES/CC-BY-4.0.url.txt",
        )
    )
    for relative in EXPECTED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".mp4") or relative.endswith(".png"):
            path.write_bytes(relative.encode("ascii"))
        elif relative == "NOTICE.txt":
            path.write_text(
                "FlyGym 2.1.0\nCopyright 2023-2026 The NeuroMechFly v2 Authors\n"
                "Apache License 2.0\nMaleCNS v1.0\nCC BY 4.0\n"
                "simulated and uncalibrated. Changes were made.\n",
                encoding="utf-8",
            )
        elif relative in {"index.html", "replay.html"}:
            path.write_text(f"シミュレーション記録 未較正 {links}\n", encoding="utf-8")
        elif relative == "assets/replay-data.js":
            write_replay_data(root, replay_data())
        elif relative == "SOURCE_ATTRIBUTIONS.json":
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("review fixture\n", encoding="utf-8")


def test_replay_package_manifest_round_trip(tmp_path):
    make_package(tmp_path)
    build_manifest(tmp_path, "2026-09-22")
    assert verify_package(tmp_path) == []


def test_replay_package_accepts_explicit_owner_authorization(tmp_path):
    make_package(tmp_path)
    manifest = build_manifest(tmp_path, "2026-09-22", AUTHORIZED_GATE)
    assert manifest["publication_gate"] == AUTHORIZED_GATE
    assert verify_package(tmp_path) == []


def test_published_hae_package_matches_authorized_manifest():
    package = ROOT / "examples" / "hae-recorded-replay"
    manifest = json.loads((package / "PUBLIC_FILE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["publication_gate"] == AUTHORIZED_GATE
    assert verify_package(package) == []
    for path in package.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".cmd", ".html", ".js", ".json", ".md", ".txt"}:
            assert b"\r\n" not in path.read_bytes(), f"non-canonical CRLF in {path}"


def test_release_audit_only_allows_manifest_reviewed_binaries():
    reviewed, failures = reviewed_binary_paths(ROOT)
    assert failures == []
    assert reviewed == {
        "examples/hae-recorded-replay/assets/actual-motion.mp4",
        "examples/hae-recorded-replay/assets/pose-045010.png",
        "examples/hae-recorded-replay/assets/replay-data.js",
    }


def test_replay_package_rejects_source_recording_and_raw_input(tmp_path):
    make_package(tmp_path)
    (tmp_path / "observation.npz").write_bytes(b"source recording")
    with pytest.raises(ValueError, match="unexpected"):
        build_manifest(tmp_path, "2026-09-22")


def test_replay_package_rejects_absolute_path_and_stale_hash(tmp_path):
    make_package(tmp_path)
    build_manifest(tmp_path, "2026-09-22")
    windows_path = "C:" + "\\Users\\example\\secret\n"
    (tmp_path / "README.md").write_text(windows_path, encoding="utf-8")
    failures = verify_package(tmp_path)
    assert "absolute local path in README.md" in failures
    assert "hash mismatch: README.md" in failures


@pytest.mark.parametrize("payload", ["[]", "null", "true", "1", '"manifest"', "{broken"])
def test_replay_package_reports_invalid_manifest_without_raising(tmp_path, payload):
    make_package(tmp_path)
    build_manifest(tmp_path, "2026-10-02")
    (tmp_path / "PUBLIC_FILE_MANIFEST.json").write_text(payload, encoding="utf-8")
    assert verify_package(tmp_path)


@pytest.mark.parametrize("gate", [[], {}, None, 1])
def test_replay_package_reports_invalid_manifest_gate_without_raising(tmp_path, gate):
    make_package(tmp_path)
    manifest = build_manifest(tmp_path, "2026-10-02")
    manifest["publication_gate"] = gate
    (tmp_path / "PUBLIC_FILE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "publication gate is not a recognized explicit owner decision" in verify_package(tmp_path)


@pytest.mark.parametrize("relative", ["PUBLIC_FILE_MANIFEST.json", "NOTICE.txt", "index.html", "replay.html", "assets/replay-data.js", "SOURCE_ATTRIBUTIONS.json"])
def test_replay_package_reports_non_utf8_text_without_raising(tmp_path, relative):
    make_package(tmp_path)
    build_manifest(tmp_path, "2026-10-02")
    (tmp_path / relative).write_bytes(b"\xff")
    assert any(relative in failure for failure in verify_package(tmp_path))


@pytest.mark.parametrize("payload", [[], None, "provenance", 0])
def test_replay_package_reports_non_object_provenance(tmp_path, payload):
    make_package(tmp_path)
    write_replay_data(tmp_path, replay_data(), payload)
    build_manifest(tmp_path, "2026-10-02")
    assert "REPLAY_PROVENANCE must be a JSON object" in verify_package(tmp_path)


@pytest.mark.parametrize("key", ["time_s", "contacts", "muscle_force", "cum_spikes", "cum_sensory", "root_xyz"])
def test_replay_package_rejects_truncated_series_with_fresh_manifest(tmp_path, key):
    make_package(tmp_path)
    data = replay_data()
    data[key].pop()
    write_replay_data(tmp_path, data)
    build_manifest(tmp_path, "2026-10-02")
    assert f"REPLAY_DATA.{key} must contain meta.frames values" in verify_package(tmp_path)


@pytest.mark.parametrize("key,value,message", [
    ("frames", True, "meta.frames must be a positive integer"),
    ("frames", 10**400, "meta.frames must be a positive integer"),
    ("fps", 0, "fps and duration_s"),
    ("fps", float("inf"), "fps and duration_s"),
    ("duration_s", 0.5, "frame count, fps and duration_s are inconsistent"),
    ("feet", ["lf"] * 6, "meta.feet"),
    ("feet", [[], "lm", "lh", "rf", "rm", "rh"], "meta.feet"),
    ("source_sha256", "missing", "meta.source_sha256"),
    ("video_sha256", "0" * 64, "video_sha256 does not match"),
    ("pose_interpolation", True, "meta.pose_interpolation"),
    ("walking_passed", True, "meta.walking_passed"),
    ("force_units", "", "meta.force_units"),
    ("total_spikes", True, "meta.total_spikes"),
])
def test_replay_package_rejects_invalid_metadata_with_fresh_manifest(tmp_path, key, value, message):
    make_package(tmp_path)
    data = replay_data()
    data["meta"][key] = value
    write_replay_data(tmp_path, data)
    build_manifest(tmp_path, "2026-10-02")
    assert any(message in failure for failure in verify_package(tmp_path))


@pytest.mark.parametrize("key,value,message", [
    ("time_s", [0.001, 0.034, float("nan")], "time_s must contain finite"),
    ("time_s", [0.001, 0.034, 10**400], "time_s must contain finite"),
    ("time_s", [0.001, 0.034, 0.034], "time_s must increase strictly"),
    ("time_s", [0.001, 0.034, 0.2], "time_s must increase strictly"),
    ("time_s", [0.001, 0.07, 0.09], "time_s is not synchronized"),
    ("contacts", [[1, 0]] * 3, "contacts must contain six"),
    ("contacts", [[2, 0, 0, 0, 0, 0]] * 3, "contacts must contain six"),
    ("root_xyz", [[0.0, float("inf"), 1.0]] * 3, "root_xyz must contain three finite"),
    ("muscle_force", [0.0, -1.0, 0.0], "muscle_force must contain finite nonnegative"),
    ("cum_spikes", [0, 4, 2], "cum_spikes must be nondecreasing"),
    ("cum_spikes", [0, 2, 6], "cum_spikes must be nondecreasing"),
    ("cum_sensory", [0, True, 2], "cum_sensory must contain nonnegative integers"),
    ("cum_sensory", [0, 1, 2.0], "cum_sensory must contain nonnegative integers"),
])
def test_replay_package_rejects_invalid_samples_with_fresh_manifest(tmp_path, key, value, message):
    make_package(tmp_path)
    data = replay_data()
    data[key] = value
    write_replay_data(tmp_path, data)
    build_manifest(tmp_path, "2026-10-02")
    assert any(message in failure for failure in verify_package(tmp_path))


def test_replay_package_rejects_duplicate_data_assignment(tmp_path):
    make_package(tmp_path)
    data_path = tmp_path / "assets/replay-data.js"
    data_path.write_text(data_path.read_text() + "\nwindow.REPLAY_DATA = {};\n", encoding="utf-8")
    build_manifest(tmp_path, "2026-10-02")
    assert "replay-data.js must contain exactly one JSON REPLAY_DATA assignment" in verify_package(tmp_path)


def test_replay_package_accepts_json_delimiters_in_strings(tmp_path):
    make_package(tmp_path)
    data = replay_data()
    data["meta"]["run"] = "simulated }; run"
    write_replay_data(tmp_path, data)
    build_manifest(tmp_path, "2026-10-02")
    assert verify_package(tmp_path) == []


@pytest.mark.parametrize("assignment", [
    "window.REPLAY_DATA = {broken};\n",
    "window.REPLAY_DATA = [] ;\n",
    "window.REPLAY_DATA = null;\n",
    "window.REPLAY_DATA = {}\n",
])
def test_replay_package_reports_invalid_data_assignment(tmp_path, assignment):
    make_package(tmp_path)
    (tmp_path / "assets/replay-data.js").write_text(
        "window.REPLAY_PROVENANCE = " + json.dumps(PROVENANCE) + ";\n" + assignment,
        encoding="utf-8",
    )
    build_manifest(tmp_path, "2026-10-02")
    assert any("REPLAY_DATA" in failure for failure in verify_package(tmp_path))


def test_replay_package_rejects_conflicting_provenance_gate(tmp_path):
    make_package(tmp_path)
    write_replay_data(tmp_path, replay_data(), {**PROVENANCE, "publication_gate": AUTHORIZED_GATE})
    build_manifest(tmp_path, "2026-10-02")
    assert "replay provenance publication_gate must match the manifest" in verify_package(tmp_path)


def test_replay_package_rejects_video_replacement_with_fresh_manifest(tmp_path):
    make_package(tmp_path)
    (tmp_path / "assets/actual-motion.mp4").write_bytes(b"different recorded video")
    build_manifest(tmp_path, "2026-10-02")
    assert "REPLAY_DATA video_sha256 does not match assets/actual-motion.mp4" in verify_package(tmp_path)


def test_replay_package_rejects_invalid_attributions_json(tmp_path):
    make_package(tmp_path)
    (tmp_path / "SOURCE_ATTRIBUTIONS.json").write_text("[]\n", encoding="utf-8")
    build_manifest(tmp_path, "2026-10-02")
    assert "SOURCE_ATTRIBUTIONS.json must be a JSON object" in verify_package(tmp_path)
