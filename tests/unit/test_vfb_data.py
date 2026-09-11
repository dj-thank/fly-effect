"""Synthetic protocol tests, not animal data or live biological validation."""
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from organism_core import vfb_data as v
from organism_core import vfb_enrichment as e


def node(i):
    return {"cursor": i, "labels": ["Individual", "Neuron"],
            "properties": {"short_form": f"VFB_{i}", "unknown_future_field": ["kept"]}}


def edge(i, accession="1", target="male_cns_v1_0", vfb="VFB_1", labels=None):
    return {"cursor": i, "source": i + 1, "target": 1000,
            "type": "database_cross_reference", "source_id": vfb, "target_id": target,
            "properties": {"accession": [accession]},
            "source_labels": labels or ["Individual", "Neuron"],
            "source_properties": {"short_form": vfb},
            "target_labels": ["Individual", "Site"], "target_properties": {"short_form": target}}


class FakeClient:
    endpoint = v.ENDPOINTS[0]

    def __init__(self, scope="all", nodes=None, edges=None):
        self.scope = scope
        self.data = {"nodes": [node(0), node(7)] if nodes is None else nodes,
                     "edges": [edge(0)] if edges is None else edges}
        self.calls = []
        self.catalogue = [{"id": "fixture_only", "properties": {"version": "test"}}]

    def query(self, query, parameters=None):
        self.calls.append((query, parameters))
        if query == v.CATALOGUE_QUERY:
            return self.catalogue
        for phase in v.SCOPES[self.scope]:
            if query == v.query_for(self.scope, phase, count=True):
                return [{"count": len(self.data[phase])}]
            if query == v.query_for(self.scope, phase):
                return [r for r in self.data[phase] if r["cursor"] > parameters["cursor"]][:parameters["limit"]]
        raise AssertionError(query)


def exported(tmp_path, rows=None, scope="xrefs"):
    out = tmp_path / "export"
    c = FakeClient(scope, edges=rows)
    state = v.export(c, out, scope=scope, source_db="male_cns_v1_0" if scope == "xrefs" else "")
    assert state["status"] == "complete_count_checked"
    return out, state


def test_all_records_and_unknown_fields_preserved(tmp_path):
    c = FakeClient()
    state = v.export(c, tmp_path / "all", scope="all", page_size=1)
    assert state["status"] == "complete_count_checked"
    assert state["counts"] == {"nodes": 2, "edges": 1}
    pages = list(v.verified_pages(tmp_path / "all"))
    assert pages[0][1][0]["properties"]["unknown_future_field"] == ["kept"]
    assert not state["snapshot_is_atomic"]
    assert not state["simulation_graph_modified"]
    assert not state["biological_validation"]


@pytest.mark.parametrize("scope", list(v.SCOPES))
def test_each_scope_and_no_hidden_threshold(scope, tmp_path):
    c = FakeClient(scope)
    state = v.export(c, tmp_path / scope, scope=scope, source_db="male_cns_v1_0")
    assert state["status"] == "complete_count_checked"
    for query, parameters in c.calls:
        assert "count >" not in query and "count >=" not in query
        assert "NOT" not in query  # No silently omitted deprecated/source populations.
        assert all(word not in query for word in ("DELETE ", "SET ", "CREATE ", "MERGE "))


@pytest.mark.parametrize("budget", [{"max_pages": 1}, {"max_rows": 1}, {"max_bytes": 1}])
def test_budget_resume_keeps_pages(tmp_path, budget):
    out = tmp_path / "resume"
    c = FakeClient()
    state = v.export(c, out, scope="all", page_size=1, **budget)
    assert state["status"] == "incomplete_budget"
    prior = [p["sha256"] for p in state["pages"]]
    state = v.export(c, out, scope="all", resume=True)
    assert state["status"] == "complete_count_checked"
    assert prior == [p["sha256"] for p in state["pages"]][:len(prior)]
    assert len(list(v.verified_pages(out))) >= 2


@pytest.mark.parametrize("value", [True, 0, -1, 1.1, "5", None])
def test_bad_integer_budgets(value, tmp_path):
    with pytest.raises(v.VFBError):
        v.export(FakeClient(), tmp_path / "new", max_pages=value)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_bad_wall_budget(value, tmp_path):
    with pytest.raises(v.VFBError):
        v.export(FakeClient(), tmp_path / "new", wall_seconds=value)


def test_empty_population_has_no_completion_fraction(tmp_path):
    out, state = exported(tmp_path, rows=[])
    summary, rows = e.join_exact_ids([], out, "male_cns_v1_0")
    assert state["scope_empty"] and summary["matched_fraction"] is None and rows == []


@pytest.mark.parametrize("bad", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'{', b'\xff'])
def test_strict_json(bad):
    with pytest.raises(v.VFBError):
        v.decode(bad)


@pytest.mark.parametrize("rows", [
    [node(3), node(3)], [node(3), node(2)], [dict(node(0), cursor=True)],
    [dict(node(0), labels="Neuron")], [dict(node(0), properties=[])],
    [dict(node(0), properties={"x": float("nan")})],
])
def test_invalid_page_is_never_written(rows, tmp_path):
    c = FakeClient(nodes=rows)
    # Return supplied invalid order rather than mimic sorting.
    original = c.query
    c.query = lambda q, p=None: rows if q == v.query_for("all", "nodes") else original(q, p)
    state = v.export(c, tmp_path / "bad", scope="all")
    assert state["status"] == "failed"
    assert not state["pages"]


def test_source_grows_resume_rejected(tmp_path):
    c = FakeClient()
    out = tmp_path / "growth"
    v.export(c, out, scope="all", max_pages=1)
    c.data["nodes"].append(node(100))
    state = v.export(c, out, scope="all", resume=True)
    assert state["status"] == "failed" and "source changed" in state["error"]


def test_changed_catalogue_same_count_rejected(tmp_path):
    c = FakeClient()
    out = tmp_path / "version"
    v.export(c, out, scope="all", max_pages=1)
    c.catalogue[0]["properties"]["version"] = "new"
    state = v.export(c, out, scope="all", resume=True)
    assert state["status"] == "failed"


def test_source_count_mismatch_not_success(tmp_path):
    c = FakeClient(nodes=[])
    original = c.query
    c.query = lambda q, p=None: [{"count": 1}] if q == v.query_for("all", "nodes", count=True) else original(q, p)
    state = v.export(c, tmp_path / "missing", scope="all")
    assert state["status"] == "failed"
    assert "enumeration" in state["error"]


def test_transport_failure_has_failure_receipt(tmp_path):
    c = FakeClient()
    c.query = lambda *_: (_ for _ in ()).throw(v.VFBError("network unavailable"))
    state = v.export(c, tmp_path / "failed", scope="all")
    assert state["status"] == "failed"
    assert v.read_receipt(tmp_path / "failed")["counts"] == {"nodes": 0, "edges": 0}


def test_fresh_workspace_refuses_overwrite(tmp_path):
    out, _ = exported(tmp_path)
    before = (out / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        v.export(FakeClient(), out)
    assert (out / "receipt.json").read_bytes() == before


@pytest.mark.parametrize("mutate", ["hash", "path", "duplicate", "claim", "accounting", "phase"])
def test_tampered_export_rejected(tmp_path, mutate):
    out, state = exported(tmp_path)
    if mutate == "hash":
        (out / state["pages"][0]["file"]).write_text("{}\n")
    elif mutate == "path":
        state["pages"][0]["file"] = "../escape.jsonl"
    elif mutate == "duplicate":
        state["pages"].append(state["pages"][0])
    elif mutate == "claim":
        state["completed_phases"] = []
    elif mutate == "accounting":
        state["counts_after"]["edges"] = 0
    else:
        state["counts"]["nodes"] = 1
    v.atomic_json(out / "receipt.json", state)
    with pytest.raises((v.VFBError, OSError)):
        list(v.verified_pages(out))


def test_existing_uncommitted_page_not_overwritten(tmp_path):
    c = FakeClient()
    out = tmp_path / "crash"
    state = v.export(c, out, scope="all", max_bytes=1)
    assert not state["pages"]
    (out / "nodes-00000000.jsonl").write_bytes(b"uncommitted")
    state = v.export(c, out, scope="all", resume=True)
    assert state["status"] == "failed"
    assert (out / "nodes-00000000.jsonl").read_bytes() == b"uncommitted"


def test_summary_distinguishes_related_from_synapses(tmp_path):
    rows = [dict(edge(0), type="synapsed_to", properties={"count": 1}),
            dict(edge(1), type="Related", properties={"count": 1})]
    out, _ = exported(tmp_path, rows, scope="all")
    summary = v.summarize(out)
    assert summary["relationship_rows_by_type"] == {"synapsed_to": 1, "Related": 1}
    assert summary["related_edges_are_not_additional_synapses"]


def test_id_matching_and_big_integers_no_precision_loss(tmp_path):
    big = 2**63 + 17
    out, _ = exported(tmp_path, [edge(0, str(big))])
    summary, rows = e.join_exact_ids([big, 4], out, "male_cns_v1_0")
    assert summary["counts"] == dict(matched=1, unmatched=1, ambiguous=0, excluded=0)
    assert rows[0]["body_id"] == str(big) and rows[1]["status"] == "unmatched"


@pytest.mark.parametrize("value", [1.0, True, -1, "1.0", "１", None])
def test_float_bool_fuzzy_ids_rejected(value):
    with pytest.raises(v.VFBError):
        e.exact_id(value)


def test_source_collision_refused(tmp_path):
    out, _ = exported(tmp_path)
    with pytest.raises(v.VFBError):
        e.join_exact_ids([1], out, "flywire783")


def test_invalid_cross_source_row_refused(tmp_path):
    out, _ = exported(tmp_path, [edge(0, target="other_version")])
    with pytest.raises(v.VFBError):
        e.join_exact_ids([1], out, "male_cns_v1_0")


def test_one_to_many_is_ambiguous(tmp_path):
    out, _ = exported(tmp_path, [edge(0), edge(1, vfb="VFB_2")])
    summary, rows = e.join_exact_ids([1], out, "male_cns_v1_0")
    assert summary["counts"]["ambiguous"] == 1 and len(rows[0]["candidates"]) == 2


def test_many_to_one_is_ambiguous_for_both(tmp_path):
    out, _ = exported(tmp_path, [edge(0), edge(1, accession="2")])
    summary, _ = e.join_exact_ids([1, 2], out, "male_cns_v1_0")
    assert summary["counts"]["ambiguous"] == 2


def test_duplicate_equivalent_xref_evidence_not_two_neurons(tmp_path):
    out, _ = exported(tmp_path, [edge(0), edge(1)])
    summary, rows = e.join_exact_ids([1], out, "male_cns_v1_0")
    assert summary["counts"]["matched"] == 1
    assert len(rows[0]["candidates"][0]["xref_evidence"]) == 2


@pytest.mark.parametrize("labels", [["Individual", "Neuron", "Deprecated"], ["Class", "Neuron"]])
def test_deprecated_or_nonindividual_excluded(tmp_path, labels):
    out, _ = exported(tmp_path, [edge(0, labels=labels)])
    summary, _ = e.join_exact_ids([1], out, "male_cns_v1_0")
    assert summary["counts"]["excluded"] == 1


def test_deprecated_source_not_accepted(tmp_path):
    row = edge(0)
    row["target_properties"]["is_obsolete"] = [True]
    out, _ = exported(tmp_path, [row])
    summary, _ = e.join_exact_ids([1], out, "male_cns_v1_0")
    assert summary["counts"]["excluded"] == 1


def test_leading_zero_id_not_fuzzily_joined(tmp_path):
    out, _ = exported(tmp_path, [edge(0, "001")])
    summary, _ = e.join_exact_ids([1], out, "male_cns_v1_0")
    assert summary["counts"]["unmatched"] == 1


def test_original_graph_lock_required_before_enrichment(tmp_path):
    p = tmp_path / "body_ids.npy"
    p.write_bytes(b"wrong-original-file")
    with pytest.raises(v.VFBError, match="graph lock"):
        e.enrich_locked_graph(tmp_path, p, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


class Response(io.BytesIO):
    pass


def response_for(rows=None):
    return v.canonical({"errors": [], "results": [{"columns": ["count"], "data": [{"row": [1]}] if rows is None else rows}]})


def mocked_client(responses, monkeypatch):
    c = v.PDBClient()
    seq = iter(responses)
    monkeypatch.setattr(v.time, "sleep", lambda _: None)
    def open_request(*args, **kwargs):
        result = next(seq)
        if isinstance(result, Exception):
            raise result
        return Response(result)
    monkeypatch.setattr(c.opener, "open", open_request)
    return c


def test_current_then_legacy_endpoint_fallback(monkeypatch):
    error = HTTPError(v.ENDPOINTS[0], 404, "missing", {}, None)
    c = mocked_client([error, response_for()], monkeypatch)
    assert c.query("MATCH (n) RETURN count(n) AS count") == [{"count": 1}]
    assert c.path == "/db/data/transaction/commit" and c.requests == 2


@pytest.mark.parametrize("raw", [
    b'{"errors":[{"code":"bad"}],"results":[]}',
    b'{"results":[]}', b'{"errors":[],"results":[null]}',
    b'{"errors":[],"results":[{"columns":["a","a"],"data":[]}]}',
    b'{"errors":[],"results":[{"columns":["a"],"data":[{"row":[]}]}]}',
])
def test_malformed_api_rejected(raw, monkeypatch):
    with pytest.raises(v.VFBError):
        mocked_client([raw], monkeypatch).query("MATCH (n) RETURN n")


def test_bounded_transport_retries(monkeypatch):
    c = mocked_client([URLError("offline")] * 3, monkeypatch)
    with pytest.raises(v.VFBError):
        c.query("MATCH (n) RETURN n")
    assert c.requests == 3


def test_oversized_response_rejected(monkeypatch):
    c = mocked_client([b" " * 100], monkeypatch)
    c.max_bytes = 20
    with pytest.raises(v.VFBError):
        c.query("MATCH (n) RETURN n")


@pytest.mark.parametrize("url", ["http://pdb.virtualflybrain.org", "https://evil.test", "file:///etc/passwd",
                                  "https://pdb.virtualflybrain.org@evil.test", "https://pdb.virtualflybrain.org/path"])
def test_endpoint_allowlist(url):
    with pytest.raises(v.VFBError):
        v.PDBClient(url)


def test_redirect_and_write_query_refused():
    with pytest.raises(v.VFBError):
        v._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.test")
    with pytest.raises(v.VFBError):
        v.PDBClient().query("CREATE (n)")


def test_incomplete_export_cannot_be_used_as_full_population(tmp_path):
    out = tmp_path / "partial"
    v.export(FakeClient("xrefs"), out, scope="xrefs", source_db="male_cns_v1_0", max_bytes=1)
    with pytest.raises(v.VFBError):
        e.join_exact_ids([1], out, "male_cns_v1_0")
