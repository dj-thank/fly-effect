"""Read-only, restartable VFB PDB acquisition; never modifies the simulated graph.

Run ``python -m organism_core.vfb_data --help``. Exported metadata is third-party
content and belongs in work/, not in Git or public CI artifacts.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import tempfile
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ENDPOINTS = ("https://pdb.v4.virtualflybrain.org", "https://pdb.virtualflybrain.org")
SCHEMA = 1
SCOPES = {
    "catalogue": {
        "nodes": "n:DataSet OR n:Site OR n:License OR n:Template",
        "edges": "s:DataSet OR s:Site OR s:License OR s:Template",
    },
    "body": {"edges": "s:Class AND s:Neuron AND t:Class AND (t:Muscle OR t:Sense_organ)"},
    "xrefs": {"edges": "type(r) = 'database_cross_reference' AND t.short_form = $source_db"},
    "all": {"nodes": "true", "edges": "true"},
}
CATALOGUE_QUERY = (
    "MATCH (n:DataSet) RETURN n.short_form AS id, properties(n) AS properties "
    "ORDER BY id"
)


class VFBError(ValueError):
    """Invalid data, provenance or remote response; never a biological failure."""


class BudgetExceeded(VFBError):
    """Acquisition incomplete under the explicitly chosen execution budget."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise VFBError(f"{name} must be an integer >= {minimum}")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VFBError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _float(text):
    result = float(text)
    if not math.isfinite(result):
        raise VFBError("non-finite JSON number")
    return result


def decode(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_float=_float,
                          parse_constant=lambda _: (_ for _ in ()).throw(VFBError("JSON constant")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise VFBError("invalid JSON") from exc


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise VFBError("value is not finite JSON") from exc


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".vfb-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise VFBError("redirect refused; use a documented HTTPS PDB endpoint")


class PDBClient:
    """Fixed public VFB endpoint, finite retry/response/socket limits, no writes.

    Public credentials match VFB_connect/default_servers.py. Socket timeout is
    not an OS wall deadline or a server-side query cancellation guarantee.
    """
    def __init__(self, endpoint=ENDPOINTS[0], timeout=30.0, max_bytes=8 * 1024 * 1024):
        if endpoint not in ENDPOINTS:
            raise VFBError("endpoint must be one of the documented HTTPS PDB hosts")
        if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
            raise VFBError("invalid socket timeout")
        self.endpoint, self.timeout = endpoint, timeout
        self.max_bytes = _integer(max_bytes, "response byte limit", 1)
        self.path = "/db/neo4j/tx/commit"
        self.opener = build_opener(_NoRedirect())
        self.requests = 0

    def query(self, statement, parameters=None):
        # Only fixed MATCH/RETURN queries are used by this module. This public
        # interface is deliberately not exposed as arbitrary CLI Cypher.
        if not statement.startswith("MATCH "):
            raise VFBError("only fixed read queries are supported")
        payload = canonical({"statements": [{"statement": statement,
                                              "parameters": parameters or {}}]})
        for attempt in range(3):
            request = Request(self.endpoint + self.path, payload, headers={
                "Content-Type": "application/json", "Accept": "application/json",
                "User-Agent": "fly-effect-vfb-research/1",
                "Authorization": "Basic " + base64.b64encode(b"neo4j:vfb").decode("ascii"),
            })
            try:
                self.requests += 1
                with self.opener.open(request, timeout=self.timeout) as response:
                    raw = response.read(self.max_bytes + 1)
                if len(raw) > self.max_bytes:
                    raise VFBError("remote response exceeds byte limit; reduce page size")
                data = decode(raw)
                if not isinstance(data, dict) or data.get("errors") != []:
                    raise VFBError("Neo4j error or missing errors field")
                results = data.get("results")
                if not isinstance(results, list) or len(results) != 1:
                    raise VFBError("expected one Neo4j result")
                result = results[0]
                if not isinstance(result, dict):
                    raise VFBError("malformed Neo4j result object")
                columns, rows = result.get("columns"), result.get("data")
                if (not isinstance(columns, list) or not all(isinstance(c, str) for c in columns)
                        or len(set(columns)) != len(columns) or not isinstance(rows, list)):
                    raise VFBError("malformed Neo4j result")
                output = []
                for row in rows:
                    values = row.get("row") if isinstance(row, dict) else None
                    if not isinstance(values, list) or len(values) != len(columns):
                        raise VFBError("malformed Neo4j row")
                    output.append(dict(zip(columns, values)))
                return output
            except HTTPError as exc:
                if exc.code == 404 and self.path == "/db/neo4j/tx/commit":
                    self.path = "/db/data/transaction/commit"
                    continue
                if exc.code not in (429, 502, 503, 504) or attempt == 2:
                    raise VFBError(f"PDB HTTP {exc.code}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise VFBError(f"PDB transport failed: {type(exc).__name__}") from exc
            time.sleep(0.25 * (attempt + 1))
        raise VFBError("PDB request attempts exhausted")


def query_for(scope, phase, *, count=False):
    try:
        predicate = SCOPES[scope][phase]
    except KeyError as exc:
        raise VFBError("unknown scope or phase") from exc
    var = "n" if phase == "nodes" else "r"
    match = "MATCH (n)" if phase == "nodes" else "MATCH (s)-[r]->(t)"
    where = f" WHERE ({predicate})"
    if count:
        return match + where + " RETURN count(*) AS count"
    where += f" AND id({var}) > $cursor"
    if phase == "nodes":
        fields = "id(n) AS cursor, labels(n) AS labels, properties(n) AS properties"
    else:
        fields = ("id(r) AS cursor, id(s) AS source, id(t) AS target, type(r) AS type, "
                  "properties(r) AS properties, s.short_form AS source_id, t.short_form AS target_id")
        if scope in ("body", "xrefs", "catalogue"):
            fields += (", labels(s) AS source_labels, properties(s) AS source_properties, "
                       "labels(t) AS target_labels, properties(t) AS target_properties")
    return match + where + " RETURN " + fields + " ORDER BY cursor LIMIT $limit"


def _count(client, scope, phase, source_db):
    rows = client.query(query_for(scope, phase, count=True), {"source_db": source_db})
    if len(rows) != 1 or set(rows[0]) != {"count"}:
        raise VFBError("invalid count response")
    return _integer(rows[0]["count"], "remote count")


def validate_rows(rows, phase, cursor):
    if not isinstance(rows, list):
        raise VFBError("page is not a row list")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("properties"), dict):
            raise VFBError("row properties missing")
        value = _integer(row.get("cursor"), "record ID")
        if value <= cursor:
            raise VFBError("nonmonotonic or duplicate record ID")
        cursor = value
        if phase == "nodes":
            if not isinstance(row.get("labels"), list) or not all(isinstance(v, str) for v in row["labels"]):
                raise VFBError("invalid labels")
        else:
            _integer(row.get("source"), "source ID")
            _integer(row.get("target"), "target ID")
            if not isinstance(row.get("type"), str) or not row["type"]:
                raise VFBError("missing relationship type")
        canonical(row)  # Preserve unknown fields; reject non-finite or unserializable data.
    return cursor


def read_receipt(workspace):
    path = Path(workspace) / "receipt.json"
    if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
        raise VFBError("unsafe or oversized receipt")
    state = decode(path.read_bytes())
    if (not isinstance(state, dict) or type(state.get("schema")) is not int
            or state["schema"] != SCHEMA or state.get("scope") not in SCOPES):
        raise VFBError("unsupported receipt")
    scope = state["scope"]
    phases = set(SCOPES[scope])
    if state.get("endpoint") not in ENDPOINTS or not isinstance(state.get("source_db"), str):
        raise VFBError("invalid export source")
    for key in ("counts", "cursors"):
        if not isinstance(state.get(key), dict) or set(state[key]) != phases:
            raise VFBError("invalid phase accounting")
        for value in state[key].values():
            _integer(value, key, -1 if key == "cursors" else 0)
    done = state.get("completed_phases")
    if not isinstance(done, list) or not all(isinstance(p, str) for p in done):
        raise VFBError("invalid phase completion")
    if len(set(done)) != len(done) or not set(done) <= phases:
        raise VFBError("invalid phase completion")
    if state.get("status") == "complete_count_checked":
        if (set(done) != phases or state.get("counts_before") != state["counts"]
                or state.get("counts_after") != state["counts"]):
            raise VFBError("unsupported complete-export claim")
    return state


def verified_pages(workspace, state=None):
    """Yield checksum-checked pages without trusting filenames or stored counts."""
    workspace = Path(workspace)
    state = read_receipt(workspace) if state is None else state
    cursors = {phase: -1 for phase in SCOPES[state["scope"]]}
    counts = {phase: 0 for phase in cursors}
    names = set()
    pages = state.get("pages")
    if not isinstance(pages, list):
        raise VFBError("missing page manifest")
    for page in pages:
        if not isinstance(page, dict):
            raise VFBError("invalid page manifest entry")
        phase, name = page.get("phase"), page.get("file")
        if (phase not in cursors or not isinstance(name, str)
                or not re.fullmatch(r"(?:nodes|edges)-[0-9]{8}\.jsonl", name) or name in names):
            raise VFBError("unsafe or duplicate page manifest")
        names.add(name)
        path = workspace / name
        size = _integer(page.get("bytes"), "page bytes")
        if size > 16 * 1024 * 1024 or path.is_symlink() or path.stat().st_size != size:
            raise VFBError("unsafe page or page size mismatch")
        raw = path.read_bytes()
        if digest(raw) != page.get("sha256"):
            raise VFBError("page checksum mismatch")
        rows = [decode(line) for line in raw.splitlines()]
        cursors[phase] = validate_rows(rows, phase, cursors[phase])
        if (not rows or len(rows) != page.get("rows") or cursors[phase] != page.get("last_id")):
            raise VFBError("page accounting mismatch")
        counts[phase] += len(rows)
        yield phase, rows
    if counts != state.get("counts") or cursors != state.get("cursors"):
        raise VFBError("receipt accounting mismatch")


def export(client, workspace, *, scope="catalogue", source_db="", resume=False,
           page_size=500, max_pages=100, max_rows=100000, max_bytes=100 * 1024 * 1024,
           wall_seconds=120.0):
    """Acquire graph records, not image bytes; count checks are NOT atomic snapshots.

    Budgets apply per invocation, including a cooperative wall budget. On resume
    every committed page is rechecked and scope/count/catalogue guards must match.
    A changing server can preserve counts: this cannot prove snapshot isolation.
    """
    if scope not in SCOPES or (scope == "xrefs" and not source_db):
        raise VFBError("valid scope and explicit source_db required for xrefs")
    if not isinstance(source_db, str):
        raise VFBError("source_db must be an exact string")
    for value, name in ((page_size, "page_size"), (max_pages, "max_pages"),
                        (max_rows, "max_rows"), (max_bytes, "max_bytes")):
        _integer(value, name, 1)
    if page_size > 5000:
        raise VFBError("page_size cannot exceed 5000")
    if type(wall_seconds) not in (float, int) or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise VFBError("wall_seconds must be finite and positive")
    started = time.monotonic()
    workspace = Path(workspace)
    state = None
    if resume:
        if workspace.is_symlink():
            raise VFBError("symlink workspace refused")
        state = read_receipt(workspace)
        if (state.get("endpoint") != client.endpoint or state.get("source_db") != source_db
                or state.get("scope") != scope):
            raise VFBError("resume source/scope mismatch")
        # Failure during this verification does not bless or rewrite tampered data.
        for _ in verified_pages(workspace, state):
            pass
    else:
        workspace.mkdir(parents=True, exist_ok=False)
        state = {"schema": SCHEMA, "scope": scope, "source_db": source_db,
                 "endpoint": client.endpoint, "started_at": _now(), "pages": [],
                 "counts": {p: 0 for p in SCOPES[scope]},
                 "cursors": {p: -1 for p in SCOPES[scope]}, "completed_phases": [],
                 "status": "started", "snapshot_is_atomic": False,
                 "raw_image_bytes_acquired": 0, "simulation_graph_modified": False,
                 "license_review": "per-source-required", "biological_validation": False}
    state["status"] = "running"
    state["invocation_budget"] = dict(pages=max_pages, rows=max_rows, bytes=max_bytes, wall_seconds=wall_seconds)
    state.pop("error", None)
    def checkpoint():
        state["updated_at"] = _now()
        atomic_json(workspace / "receipt.json", state)
    def wall_check():
        if time.monotonic() - started >= wall_seconds:
            raise BudgetExceeded("cooperative wall budget exhausted")
    try:
        wall_check()
        catalogue_hash = digest(canonical(client.query(CATALOGUE_QUERY)))
        before = {}
        for phase in SCOPES[scope]:
            wall_check()
            before[phase] = _count(client, scope, phase, source_db)
        if "counts_before" in state:
            if before != state["counts_before"] or catalogue_hash != state["catalogue_sha256"]:
                raise VFBError("live source changed; start a separate export")
        else:
            state["counts_before"], state["catalogue_sha256"] = before, catalogue_hash
        queries = {p: digest(query_for(scope, p).encode()) for p in SCOPES[scope]}
        if "queries" in state and queries != state["queries"]:
            raise VFBError("query implementation changed; do not resume")
        state["queries"] = queries
        checkpoint()
        used_pages = used_rows = used_bytes = 0
        for phase in SCOPES[scope]:
            if phase in state["completed_phases"]:
                continue
            while True:
                wall_check()
                if used_pages >= max_pages or used_rows >= max_rows or used_bytes >= max_bytes:
                    raise BudgetExceeded("page/row/byte budget exhausted")
                limit = min(page_size, max_rows - used_rows)
                rows = client.query(query_for(scope, phase), {"cursor": state["cursors"][phase],
                                    "limit": limit, "source_db": source_db})
                wall_check()
                if len(rows) > limit:
                    raise VFBError("server exceeded requested row limit")
                last_id = validate_rows(rows, phase, state["cursors"][phase])
                if not rows:
                    if state["counts"][phase] != before[phase]:
                        raise VFBError("page enumeration does not match source count")
                    state["completed_phases"].append(phase)
                    checkpoint()
                    break
                raw = b"".join(canonical(row) + b"\n" for row in rows)
                if len(raw) > 16 * 1024 * 1024:
                    raise VFBError("page exceeds local page cap; reduce page_size")
                if used_bytes + len(raw) > max_bytes:
                    raise BudgetExceeded("byte budget would be exceeded")
                name = f"{phase}-{len(state['pages']):08d}.jsonl"
                # A crash may leave an uncommitted page; never silently overwrite it.
                with (workspace / name).open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                state["pages"].append({"phase": phase, "file": name, "bytes": len(raw),
                                       "rows": len(rows), "last_id": last_id, "sha256": digest(raw)})
                state["counts"][phase] += len(rows)
                state["cursors"][phase] = last_id
                if state["counts"][phase] > before[phase]:
                    raise VFBError("source count grew during export")
                used_pages += 1
                used_rows += len(rows)
                used_bytes += len(raw)
                checkpoint()
        after = {}
        for phase in SCOPES[scope]:
            wall_check()
            after[phase] = _count(client, scope, phase, source_db)
        wall_check()
        if after != before or digest(canonical(client.query(CATALOGUE_QUERY))) != catalogue_hash:
            raise VFBError("live source changed during export")
        state["counts_after"] = after
        state["status"] = "complete_count_checked"
        state["scope_empty"] = sum(after.values()) == 0
    except BudgetExceeded as exc:
        state["status"], state["error"] = "incomplete_budget", str(exc)
    except (VFBError, OSError, KeyError, TypeError) as exc:
        state["status"], state["error"] = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        state["invocation_elapsed_seconds"] = time.monotonic() - started
        checkpoint()
    return state


def summarize(workspace):
    """Produce aggregate, non-payload evidence; edge counts are graph rows, not synapses."""
    state = read_receipt(workspace)
    labels, relations = {}, {}
    assets = 0
    for phase, rows in verified_pages(workspace, state):
        for row in rows:
            if phase == "nodes":
                for label in row["labels"]:
                    labels[label] = labels.get(label, 0) + 1
            else:
                relation = row["type"]
                relations[relation] = relations.get(relation, 0) + 1
                if relation == "in_register_with":
                    assets += sum(k in row["properties"] for k in ("swc", "obj", "nrrd", "wlz"))
    return {"schema": SCHEMA, "scope": state["scope"], "status": state["status"],
            "endpoint": state["endpoint"], "source_db": state["source_db"],
            "counts": state["counts"], "labels": labels, "relationship_rows_by_type": relations,
            "registration_asset_fields": assets, "receipt_sha256": digest(canonical(state)),
            "page_hashes": [p["sha256"] for p in state["pages"]],
            "snapshot_is_atomic": False, "raw_images_downloaded": 0,
            "related_edges_are_not_additional_synapses": True,
            "simulation_graph_modified": False, "biological_validation": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("export")
    acquire.add_argument("--out", type=Path, required=True)
    acquire.add_argument("--scope", choices=SCOPES, default="catalogue")
    acquire.add_argument("--source-db", default="")
    acquire.add_argument("--endpoint", choices=ENDPOINTS, default=ENDPOINTS[0])
    acquire.add_argument("--resume", action="store_true")
    acquire.add_argument("--page-size", type=int, default=500)
    acquire.add_argument("--max-pages", type=int, default=100)
    acquire.add_argument("--max-rows", type=int, default=100000)
    acquire.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024)
    acquire.add_argument("--wall-seconds", type=float, default=120)
    check = sub.add_parser("verify")
    check.add_argument("workspace", type=Path)
    check.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            state = export(PDBClient(args.endpoint), args.out, scope=args.scope,
                           source_db=args.source_db, resume=args.resume, page_size=args.page_size,
                           max_pages=args.max_pages, max_rows=args.max_rows,
                           max_bytes=args.max_bytes, wall_seconds=args.wall_seconds)
            print(canonical({k: state.get(k) for k in ("status", "counts", "error")}).decode())
            return {"complete_count_checked": 0, "incomplete_budget": 3}.get(state["status"], 2)
        summary = summarize(args.workspace)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            if args.summary.exists():
                raise VFBError("summary destination already exists")
            atomic_json(args.summary, summary)
        print(canonical(summary).decode())
        return 0 if summary["status"] == "complete_count_checked" else 2
    except (VFBError, OSError, KeyError, TypeError) as exc:
        print(f"VFB operation failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
