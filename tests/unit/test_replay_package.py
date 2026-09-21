import json
from pathlib import Path

import pytest

from scripts.replay_package import EXPECTED_FILES, build_manifest, verify_package


PROVENANCE = {
    "recording_kind": "simulated_recorded_run",
    "calibration_status": "uncalibrated",
    "source_recording_included": False,
    "raw_inputs_included": False,
    "changes_made": True,
}


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
            path.write_text(
                "window.REPLAY_PROVENANCE = "
                + json.dumps(PROVENANCE)
                + ";\nwindow.REPLAY_DATA = {};\n",
                encoding="utf-8",
            )
        else:
            path.write_text("review fixture\n", encoding="utf-8")


def test_replay_package_manifest_round_trip(tmp_path):
    make_package(tmp_path)
    build_manifest(tmp_path, "2026-09-22")
    assert verify_package(tmp_path) == []


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
