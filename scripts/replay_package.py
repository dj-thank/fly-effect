"""Build and verify the allowlisted HAE recorded-replay package manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath


MANIFEST_NAME = "PUBLIC_FILE_MANIFEST.json"
BLOCKED_GATE = "PUBLIC_DISTRIBUTION_BLOCKED_PENDING_OWNER_AUTHORIZATION"
AUTHORIZED_GATE = "PUBLIC_DISTRIBUTION_AUTHORIZED_BY_OWNER_20260922"
PUBLICATION_GATES = {BLOCKED_GATE, AUTHORIZED_GATE}
EXPECTED_FILES = {
    "README.md",
    "PROVENANCE.md",
    "NOTICE.txt",
    "SOURCE_ATTRIBUTIONS.json",
    "index.html",
    "launch.cmd",
    "replay.html",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/CC-BY-4.0.url.txt",
    "assets/actual-motion.mp4",
    "assets/pose-045010.png",
    "assets/replay-data.js",
    "assets/replay.js",
}
MIXED_LICENSE_FILES = {
    "assets/actual-motion.mp4",
    "assets/pose-045010.png",
    "assets/replay-data.js",
}
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".parquet", ".bin", ".csv", ".xml"}
TEXT_SUFFIXES = {".cmd", ".html", ".js", ".json", ".md", ".txt"}
LOCAL_PATH = re.compile(
    r"(?i)(?:(?<![a-z0-9])[a-z]:[\\/]|file://|\\\\\?\\|/home/|/Users/)"
)
SECRET = re.compile(
    r"(?i)(?:ghp_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|"
    r"sk-(?:proj-)?[a-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)"
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REPLAY_SERIES = ("time_s", "contacts", "muscle_force", "cum_spikes", "cum_sensory", "root_xyz")
FEET = {"lf", "lm", "lh", "rf", "rm", "rh"}


def _json_object(text: str, label: str, failures: list[str]) -> dict | None:
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as exc:
        failures.append(f"invalid {label}: {exc}")
        return None
    if not isinstance(value, dict):
        failures.append(f"{label} must be a JSON object")
        return None
    return value


def _json_assignment(text: str, name: str, failures: list[str]) -> dict | None:
    matches = list(re.finditer(rf"window\.{re.escape(name)}\s*=\s*", text))
    if len(matches) != 1:
        failures.append(f"replay-data.js must contain exactly one JSON {name} assignment")
        return None
    payload = text[matches[0].end():]
    try:
        value, end = json.JSONDecoder().raw_decode(payload)
    except (ValueError, RecursionError) as exc:
        failures.append(f"invalid {name} JSON: {exc}")
        return None
    if not payload[end:].lstrip().startswith(";"):
        failures.append(f"{name} must be a JSON literal terminated by a semicolon")
        return None
    if not isinstance(value, dict):
        failures.append(f"{name} must be a JSON object")
        return None
    return value


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _nonnegative_int(value: object) -> bool:
    # The browser stores these counts as JavaScript numbers.
    return type(value) is int and 0 <= value <= 2**53 - 1


def _verify_replay_data(data: dict, video_hash: str | None, failures: list[str]) -> None:
    meta = data.get("meta")
    if not isinstance(meta, dict):
        failures.append("REPLAY_DATA.meta must be an object")
        return
    frames, fps, duration = meta.get("frames"), meta.get("fps"), meta.get("duration_s")
    valid_frames = _nonnegative_int(frames) and frames > 0
    valid_clock = _finite_number(fps) and fps > 0 and _finite_number(duration) and duration > 0
    if not valid_frames:
        failures.append("REPLAY_DATA.meta.frames must be a positive integer")
    if not valid_clock:
        failures.append("REPLAY_DATA.meta.fps and duration_s must be finite positive numbers")
    if valid_frames and valid_clock and not math.isclose(
        duration, frames / fps, rel_tol=0, abs_tol=0.5 / fps + 1e-9
    ):
        failures.append("REPLAY_DATA frame count, fps and duration_s are inconsistent")
    feet = meta.get("feet")
    if (
        not isinstance(feet, list)
        or len(feet) != len(FEET)
        or any(not isinstance(foot, str) for foot in feet)
        or set(feet) != FEET
    ):
        failures.append("REPLAY_DATA.meta.feet must name each of the six feet once")
    for key in ("source_sha256", "video_sha256"):
        digest = meta.get(key)
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            failures.append(f"REPLAY_DATA.meta.{key} must be a SHA-256 digest")
    if video_hash is not None and meta.get("video_sha256") != video_hash:
        failures.append("REPLAY_DATA video_sha256 does not match assets/actual-motion.mp4")
    for key in ("pose_interpolation", "walking_passed"):
        if meta.get(key) is not False:
            failures.append(f"REPLAY_DATA.meta.{key} must be false for this recorded package")
    if not isinstance(meta.get("force_units"), str) or not meta["force_units"].strip():
        failures.append("REPLAY_DATA.meta.force_units must disclose the force units")
    if not valid_frames:
        return
    series = {}
    for key in REPLAY_SERIES:
        values = data.get(key)
        if not isinstance(values, list) or len(values) != frames:
            failures.append(f"REPLAY_DATA.{key} must contain meta.frames values")
        else:
            series[key] = values
    times = series.get("time_s")
    if times is not None:
        if not all(_finite_number(value) for value in times):
            failures.append("REPLAY_DATA.time_s must contain finite numbers")
        elif valid_clock:
            if any(value < 0 or value > duration for value in times) or any(
                b <= a for a, b in zip(times, times[1:])
            ):
                failures.append("REPLAY_DATA.time_s must increase strictly within the run duration")
            # Recorded source poses may differ slightly from nominal video time.
            # Their timestamps must still fall within one corresponding video frame.
            if any(abs(value - index / fps) >= 1 / fps + 1e-9 for index, value in enumerate(times)):
                failures.append("REPLAY_DATA.time_s is not synchronized with the video frames")
    contacts = series.get("contacts")
    if contacts is not None and any(
        not isinstance(row, list)
        or len(row) != len(FEET)
        or any(type(value) is not int or value not in (0, 1) for value in row)
        for row in contacts
    ):
        failures.append("REPLAY_DATA.contacts must contain six binary integers per frame")
    positions = series.get("root_xyz")
    if positions is not None and any(
        not isinstance(row, list)
        or len(row) != 3
        or not all(_finite_number(value) for value in row)
        for row in positions
    ):
        failures.append("REPLAY_DATA.root_xyz must contain three finite coordinates per frame")
    force = series.get("muscle_force")
    if force is not None and not all(_finite_number(value) and value >= 0 for value in force):
        failures.append("REPLAY_DATA.muscle_force must contain finite nonnegative numbers")
    for key, total_key in (("cum_spikes", "total_spikes"), ("cum_sensory", "total_sensory")):
        total, values = meta.get(total_key), series.get(key)
        if not _nonnegative_int(total):
            failures.append(f"REPLAY_DATA.meta.{total_key} must be a nonnegative integer")
        if values is not None:
            if not all(_nonnegative_int(value) for value in values):
                failures.append(f"REPLAY_DATA.{key} must contain nonnegative integers")
            elif any(b < a for a, b in zip(values, values[1:])) or (
                _nonnegative_int(total) and any(value > total for value in values)
            ):
                failures.append(f"REPLAY_DATA.{key} must be nondecreasing and not exceed {total_key}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }


def _entry(path: str, file_path: Path) -> dict[str, object]:
    if path in MIXED_LICENSE_FILES:
        classification = "MIXED_LICENSE_REVIEW_READY"
        license_name = (
            "Fly Effect MIT wrapper where applicable; FlyGym/NeuroMechFly Apache-2.0 "
            "and MaleCNS/MANC CC BY 4.0 provenance remain separate"
        )
    elif path.startswith("LICENSES/") or path in {"NOTICE.txt", "SOURCE_ATTRIBUTIONS.json"}:
        classification = "THIRD_PARTY_NOTICE"
        license_name = "Third-party notice or license reference; not relicensed as Fly Effect MIT"
    else:
        classification = "FLY_EFFECT_ORIGINAL"
        license_name = "Fly Effect MIT; linked third-party media/data remain under their own terms"
    return {
        "path": path,
        "sha256": sha256(file_path),
        "bytes": file_path.stat().st_size,
        "classification": classification,
        "license": license_name,
    }


def build_manifest(
    root: Path,
    generated_at: str,
    publication_gate: str = BLOCKED_GATE,
) -> dict[str, object]:
    root = root.resolve()
    files = package_files(root)
    unexpected = sorted(set(files) - EXPECTED_FILES)
    missing = sorted(EXPECTED_FILES - set(files))
    if unexpected or missing:
        raise ValueError(f"package inventory mismatch; missing={missing}; unexpected={unexpected}")

    manifest = {
        "schema": 2,
        "generated_at": generated_at,
        "scope": "HAE recorded-run exhibition review package",
        "publication_gate": publication_gate,
        "license_integration_status": "COMPLETE_FOR_REVIEW",
        "source_recording_included": False,
        "raw_inputs_included": False,
        "hash_scope": f"All package files except {MANIFEST_NAME}",
        "files": [_entry(path, files[path]) for path in sorted(files)],
        "excluded_from_package": [
            "observation.npz",
            "body.xml",
            "MaleCNS/MANC raw tables",
            "FlyGym/NeuroMechFly raw meshes",
            "FFmpeg/libx264 binaries",
        ],
        "integrity_note": "Hashes prove local package bytes only and do not authorize publication.",
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _valid_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in value


def verify_package(root: Path) -> list[str]:
    root = root.resolve()
    failures: list[str] = []
    files = package_files(root)
    unexpected = sorted(set(files) - EXPECTED_FILES)
    missing = sorted(EXPECTED_FILES - set(files))
    if unexpected:
        failures.append(f"unexpected package files: {unexpected}")
    if missing:
        failures.append(f"missing package files: {missing}")

    texts: dict[str, str] = {}
    for relative, path in files.items():
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.lower() in {
            "observation.npz",
            "body.xml",
        }:
            failures.append(f"forbidden source/raw input: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                failures.append(f"non-UTF-8 text file: {relative}")
                continue
            except OSError as exc:
                failures.append(f"unreadable text file: {relative}: {exc}")
                continue
            texts[relative] = text
            if LOCAL_PATH.search(text):
                failures.append(f"absolute local path in {relative}")
            if SECRET.search(text):
                failures.append(f"possible credential in {relative}")

    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        failures.append(f"missing {MANIFEST_NAME}")
        return failures
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError) as exc:
        failures.append(f"invalid {MANIFEST_NAME}: {exc}")
        return failures
    manifest = _json_object(manifest_text, MANIFEST_NAME, failures)
    if manifest is None:
        return failures

    if manifest.get("schema") != 2:
        failures.append("manifest schema must be 2")
    gate = manifest.get("publication_gate")
    if not isinstance(gate, str) or gate not in PUBLICATION_GATES:
        failures.append("publication gate is not a recognized explicit owner decision")
    if manifest.get("license_integration_status") != "COMPLETE_FOR_REVIEW":
        failures.append("license integration is not marked complete for review")
    if manifest.get("source_recording_included") is not False:
        failures.append("source recording must be excluded")
    if manifest.get("raw_inputs_included") is not False:
        failures.append("raw inputs must be excluded")

    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        failures.append("manifest files must be a list")
        entries = []
    indexed: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            failures.append("manifest contains an invalid file entry")
            continue
        relative = entry["path"]
        if not _valid_relative_path(relative):
            failures.append(f"manifest path is not relative: {relative}")
            continue
        if relative in indexed:
            failures.append(f"duplicate manifest path: {relative}")
        indexed[relative] = entry

    if set(indexed) != set(files):
        failures.append("manifest file set does not match package file set")
    for relative in sorted(set(indexed) & set(files)):
        path = files[relative]
        entry = indexed[relative]
        if entry.get("sha256") != sha256(path):
            failures.append(f"hash mismatch: {relative}")
        if entry.get("bytes") != path.stat().st_size:
            failures.append(f"size mismatch: {relative}")

    if "SOURCE_ATTRIBUTIONS.json" in texts:
        _json_object(texts["SOURCE_ATTRIBUTIONS.json"], "SOURCE_ATTRIBUTIONS.json", failures)
    notice = texts.get("NOTICE.txt", "")
    for phrase in (
        "FlyGym 2.1.0",
        "Copyright 2023-2026 The NeuroMechFly v2 Authors",
        "Apache License 2.0",
        "MaleCNS v1.0",
        "CC BY 4.0",
        "simulated",
        "uncalibrated",
        "Changes were made",
    ):
        if phrase not in notice:
            failures.append(f"NOTICE.txt missing required disclosure: {phrase}")

    required_links = (
        "NOTICE.txt",
        "PROVENANCE.md",
        "SOURCE_ATTRIBUTIONS.json",
        "LICENSES/Apache-2.0.txt",
        "LICENSES/CC-BY-4.0.url.txt",
    )
    for page_name in ("index.html", "replay.html"):
        page = texts.get(page_name, "")
        for target in required_links:
            if target not in page:
                failures.append(f"{page_name} does not link {target}")
        if "シミュレーション記録" not in page or "未較正" not in page:
            failures.append(f"{page_name} does not visibly label simulated/uncalibrated status")

    data_text = texts.get("assets/replay-data.js", "")
    provenance = _json_assignment(data_text, "REPLAY_PROVENANCE", failures)
    if provenance is not None:
        expected = {
            "recording_kind": "simulated_recorded_run",
            "calibration_status": "uncalibrated",
            "source_recording_included": False,
            "raw_inputs_included": False,
            "changes_made": True,
        }
        for key, value in expected.items():
            actual = provenance.get(key)
            if actual != value or type(actual) is not type(value):
                failures.append(f"replay provenance {key} must be {value!r}")
        if "publication_gate" in provenance and provenance["publication_gate"] != manifest.get("publication_gate"):
            failures.append("replay provenance publication_gate must match the manifest")
    data = _json_assignment(data_text, "REPLAY_DATA", failures)
    if data is not None:
        video = files.get("assets/actual-motion.mp4")
        _verify_replay_data(data, sha256(video) if video is not None else None, failures)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("package", type=Path)
    build.add_argument("--generated-at", required=True)
    build.add_argument(
        "--publication-gate",
        choices=sorted(PUBLICATION_GATES),
        default=BLOCKED_GATE,
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("package", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        manifest = build_manifest(args.package, args.generated_at, args.publication_gate)
        print(json.dumps({"status": "BUILT", "files": len(manifest["files"])}, ensure_ascii=False))
        return

    failures = verify_package(args.package)
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "PASS", "files": len(EXPECTED_FILES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
