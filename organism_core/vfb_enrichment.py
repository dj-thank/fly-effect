"""Use VFB xrefs as an evidence sidecar, never as replacement neural wiring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
from pathlib import Path
import re

from .vfb_data import VFBError, atomic_json, canonical, decode, digest, read_receipt, verified_pages

# The pinned graph is a custom derivative. Matching a source ID does not prove
# equivalence to VFB's differently filtered connection table.
GRAPH_SOURCE = {"male-cns:v1.0": "male_cns_v1_0"}


def _values(value):
    return value if isinstance(value, list) else [value]


def _deprecated(labels, properties):
    return "Deprecated" in labels or any(
        any(v is True or v == "true" for v in _values(properties.get(key, False)))
        for key in ("deprecated", "is_obsolete")
    )


def exact_id(value):
    """External numeric body IDs must not pass through a float or fuzzy label."""
    if type(value) is int and value >= 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return value
    raise VFBError("body ID must be an integer or an exact ASCII decimal string")


def join_exact_ids(body_ids, workspace, source_db):
    """Match the supplied population against a verified, complete xref export.

    Returns a derived candidate sidecar with raw labels/properties. Unknown,
    obsolete, non-neuronal and many-to-one mappings are not silently accepted.
    """
    if not isinstance(source_db, str) or not source_db:
        raise VFBError("explicit source database/version required")
    ids = [exact_id(v) for v in body_ids]
    if len(set(ids)) != len(ids):
        raise VFBError("duplicate input body IDs")
    wanted = set(ids)
    state = read_receipt(workspace)
    if (state.get("status") != "complete_count_checked" or state.get("scope") != "xrefs"
            or state.get("source_db") != source_db):
        raise VFBError("requires complete xrefs from the exact source database")
    found = defaultdict(dict)
    excluded = defaultdict(set)
    for phase, rows in verified_pages(workspace, state):
        if phase != "edges":
            raise VFBError("unexpected non-xref phase")
        for row in rows:
            if row.get("type") != "database_cross_reference" or row.get("target_id") != source_db:
                raise VFBError("cross-source/non-xref row in source-qualified export")
            accessions = row["properties"].get("accession")
            if (not isinstance(accessions, list) or not accessions
                    or not all(type(a) in (str, int) for a in accessions)):
                raise VFBError("missing or malformed xref accessions")
            # Keep every accession, not only upstream's accession[0]. Numerical
            # formatting is not normalized, so 001 does not silently become 1.
            keys = {str(a) for a in accessions} & wanted
            if not keys:
                continue
            vfb_id = row.get("source_id")
            labels, properties = row.get("source_labels"), row.get("source_properties")
            db_labels, db_properties = row.get("target_labels"), row.get("target_properties")
            if (not isinstance(vfb_id, str) or not vfb_id or not isinstance(labels, list)
                    or not all(isinstance(v, str) for v in labels) or not isinstance(properties, dict)
                    or not isinstance(db_labels, list) or not isinstance(db_properties, dict)):
                raise VFBError("xref endpoint evidence missing")
            reason = None
            if _deprecated(labels, properties) or _deprecated(db_labels, db_properties):
                reason = "deprecated"
            elif "Neuron" not in labels or "Individual" not in labels:
                reason = "not_neuronal_individual"
            for key in keys:
                if reason:
                    excluded[key].add(reason)
                    continue
                prior = found[key].get(vfb_id)
                evidence = {"vfb_id": vfb_id, "labels": labels, "properties": properties,
                            "xref_evidence": []}
                if prior is not None and (prior["labels"] != labels or prior["properties"] != properties):
                    raise VFBError("conflicting records for one VFB neuron")
                evidence = prior or evidence
                evidence["xref_evidence"].append({"record_id": row["cursor"], "properties": row["properties"]})
                found[key][vfb_id] = evidence
    reverse = defaultdict(set)
    for key, candidates in found.items():
        for vfb_id in candidates:
            reverse[vfb_id].add(key)
    counts, output = Counter(), []
    for key in ids:
        candidates = found[key]
        if len(candidates) == 1 and all(len(reverse[v]) == 1 for v in candidates):
            status = "matched"
        elif candidates:
            status = "ambiguous"
        elif excluded[key]:
            status = "excluded"
        else:
            status = "unmatched"
        counts[status] += 1
        output.append({"body_id": key, "status": status,
                       "candidates": [candidates[v] for v in sorted(candidates)],
                       "excluded_reasons": sorted(excluded[key])})
    total = len(ids)
    summary = {"source_db": source_db, "population": total,
               "counts": {s: counts[s] for s in ("matched", "unmatched", "ambiguous", "excluded")},
               "matched_fraction": counts["matched"] / total if total else None,
               "xref_receipt_sha256": digest(canonical(state)),
               "matched_is_identity_candidate_not_biological_validation": True,
               "simulation_graph_modified": False, "anatomical_motor_calibration": False}
    return summary, output


def enrich_locked_graph(workspace, body_ids_path, out):
    """Read only the hash-pinned original body-ID file; save a separate sidecar."""
    import numpy as np  # Existing project dependency; not needed for acquisition.

    lock = decode(Path(__file__).with_name("graph_lock.json").read_bytes())
    source_db = GRAPH_SOURCE.get(lock.get("dataset"))
    if not source_db:
        raise VFBError("graph release has no reviewed VFB source crosswalk")
    with Path(body_ids_path).open("rb") as stream:
        payload = stream.read(64 * 1024 * 1024 + 1)
    if len(payload) > 64 * 1024 * 1024 or digest(payload) != lock["files"]["body_ids.npy"]:
        raise VFBError("body_ids.npy does not match the pinned graph lock")
    # Verify the same bytes that will be loaded, not a separately reopened path.
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.ndim != 1 or array.dtype.kind not in "iu":
        raise VFBError("expected one-dimensional integer body IDs")
    if len(array) != lock["neurons"]:
        raise VFBError("body-ID count differs from graph lock")
    summary, rows = join_exact_ids(array.tolist(), workspace, source_db)
    summary["graph_dataset"] = lock["dataset"]
    summary["body_ids_sha256"] = digest(payload)
    summary["graph_lock_sha256"] = digest(canonical(lock))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    with (out / "neuron-evidence.jsonl").open("xb") as stream:
        stream.write(raw)
    summary["sidecar_sha256"] = digest(raw)
    atomic_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xrefs", type=Path, required=True)
    parser.add_argument("--body-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(canonical(enrich_locked_graph(args.xrefs, args.body_ids, args.out)).decode())
        return 0
    except (VFBError, OSError, KeyError, TypeError) as exc:
        print(f"VFB enrichment failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
