"""Build and verify the allowlisted HAE recorded-replay package manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
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
            if LOCAL_PATH.search(text):
                failures.append(f"absolute local path in {relative}")
            if SECRET.search(text):
                failures.append(f"possible credential in {relative}")

    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        failures.append(f"missing {MANIFEST_NAME}")
        return failures
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        failures.append(f"invalid {MANIFEST_NAME}: {exc}")
        return failures

    if manifest.get("schema") != 2:
        failures.append("manifest schema must be 2")
    if manifest.get("publication_gate") not in PUBLICATION_GATES:
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

    notice = (root / "NOTICE.txt").read_text(encoding="utf-8") if (root / "NOTICE.txt").is_file() else ""
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
        page = (root / page_name).read_text(encoding="utf-8") if (root / page_name).is_file() else ""
        for target in required_links:
            if target not in page:
                failures.append(f"{page_name} does not link {target}")
        if "シミュレーション記録" not in page or "未較正" not in page:
            failures.append(f"{page_name} does not visibly label simulated/uncalibrated status")

    data_path = root / "assets" / "replay-data.js"
    data_text = data_path.read_text(encoding="utf-8") if data_path.is_file() else ""
    match = re.search(r"window\.REPLAY_PROVENANCE = (\{.*?\});", data_text, re.DOTALL)
    if not match:
        failures.append("replay-data.js lacks machine-readable replay provenance")
    else:
        try:
            provenance = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid replay provenance JSON: {exc}")
        else:
            expected = {
                "recording_kind": "simulated_recorded_run",
                "calibration_status": "uncalibrated",
                "source_recording_included": False,
                "raw_inputs_included": False,
                "changes_made": True,
            }
            for key, value in expected.items():
                if provenance.get(key) != value:
                    failures.append(f"replay provenance {key} must be {value!r}")

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
