from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from research.circuit_bridge.__main__ import main
from research.circuit_bridge.core import (
    Boundary, BoundaryError, Circuit, Edge, Graph, NativeEndpoint, ReplayEndpoint,
    Request, Response, SurrogateEndpoint, UNITS, canonical, digest, replay_document,
)
from research.circuit_bridge.experiment import CONDITIONS, REGION, Config, fixture, run, trial


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_reject_invalid_edge_weight(value):
    with pytest.raises(ValueError):
        Edge("a", "b", value)


@pytest.mark.parametrize("ids,edges", [
    ((), ()), (("a", "a"), ()), ((1,), ()), (("",), ()),
    (("a",), (Edge("a", "b", 1),)), (("a",), ("bad",)),
])
def test_reject_invalid_graph(ids, edges):
    with pytest.raises(ValueError):
        Graph(ids, edges)


def test_graph_is_immutable_and_provenance_cannot_claim_biology():
    ids = ["a", "b"]
    edges = [Edge("a", "b", 1)]
    graph = Graph(ids, edges)
    fingerprint = graph.fingerprint
    ids.append("c")
    edges.clear()
    assert graph.ids == ("a", "b") and graph.fingerprint == fingerprint
    with pytest.raises(dataclasses.FrozenInstanceError):
        graph.ids = ("x",)
    with pytest.raises(ValueError):
        Graph(("a",), (), evidence="real_connectome")


@pytest.mark.parametrize("region", [(), ("unknown",), ("c0", "c0")])
def test_region_rejects_missing_unknown_duplicate(region):
    with pytest.raises(ValueError):
        fixture().region(region)


def test_shuffle_only_changes_internal_edges_and_keeps_degree_counts():
    graph = fixture()
    selected = set(REGION)
    changed = graph.shuffled(REGION, 9)
    internal = lambda g: [e for e in g.edges if e.pre in selected and e.post in selected]
    boundary = lambda g: [e for e in g.edges if not (e.pre in selected and e.post in selected)]
    assert boundary(graph) == boundary(changed)
    assert Counter((e.pre, e.weight) for e in internal(graph)) == Counter((e.pre, e.weight) for e in internal(changed))
    assert Counter(e.post for e in internal(graph)) == Counter(e.post for e in internal(changed))
    assert changed == graph.shuffled(REGION, 9)
    assert graph == fixture()


def test_degenerate_shuffle_does_not_fabricate_a_change():
    graph = Graph(("a",), (Edge("a", "a", 1),))
    assert graph.shuffled(("a",), 1) == graph


@pytest.mark.parametrize("config", [
    {"dt_s": 0}, {"dt_s": -1}, {"dt_s": float("nan")}, {"dt_s": 0.03},
    {"mass": 0}, {"tau_s": 0}, {"gain": float("inf")}, {"damping": -1},
    {"steps": True}, {"steps": 0}, {"steps": 100001}, {"seeds": ()},
    {"seeds": (1, 1)}, {"seeds": (-1,)}, {"seeds": (True,)},
    {"replacement": "organoid"}, {"steps": 100000, "seeds": (0, 1, 2)},
    {"mass": 0.001},
])
def test_config_rejects_invalid_or_unbounded_experiments(config):
    with pytest.raises(ValueError):
        Config(**config)


class FakeEndpoint:
    def __init__(self, change=lambda response: response):
        self.change = change

    def exchange(self, request):
        return self.change(Response.for_request(request, [0.1] * len(request.drive)))


@pytest.mark.parametrize("field,value", [
    ("step", -1), ("step", 1), ("step", False), ("dt_s", 0.2), ("dt_s", True),
    ("channel_ids", ("b", "a")), ("units", "Hz"), ("request_sha256", "stale"),
    ("values", (float("nan"), 0)), ("values", (1.01, 0)), ("values", (0,)),
    ("values", (True, 0)),
])
def test_bad_boundary_response_is_latched_and_never_reused(field, value):
    endpoint = FakeEndpoint(lambda r: dataclasses.replace(r, **{field: value}))
    boundary = Boundary(endpoint, ("a", "b"), 0.01)
    with pytest.raises(BoundaryError):
        boundary.exchange(0, (0, 0))
    assert boundary.next_step == 0 and boundary.records == []
    with pytest.raises(BoundaryError, match="latched"):
        boundary.exchange(0, (0, 0))


@pytest.mark.parametrize("step", [1, -1, True])
def test_boundary_rejects_nonconsecutive_or_boolean_step(step):
    with pytest.raises(BoundaryError):
        Boundary(FakeEndpoint(), ("a",), 0.01).exchange(step, (0,))


def test_soft_deadline_and_missing_response():
    clock = iter([0.0, 2.0])
    boundary = Boundary(FakeEndpoint(), ("a",), 0.01, 1.0, lambda: next(clock))
    with pytest.raises(BoundaryError, match="deadline"):
        boundary.exchange(0, (0,))
    with pytest.raises(BoundaryError):
        Boundary(FakeEndpoint(lambda r: None), ("a",), 0.01).exchange(0, (0,))


def test_failure_does_not_advance_neural_state():
    graph = fixture()
    boundary = Boundary(FakeEndpoint(lambda r: dataclasses.replace(r, values=(math.nan,) * 3)), REGION, 0.01)
    circuit = Circuit(graph, REGION, "replacement", boundary=boundary)
    state = circuit.state
    with pytest.raises(BoundaryError):
        circuit.advance((1, 0, 0, 0, 0, 0), 0.01)
    assert circuit.state == state and circuit.step == 0


@pytest.mark.parametrize("mode,boundary", [("bad", None), ("replacement", None),
                                           ("intact", Boundary(FakeEndpoint(), REGION, 0.01))])
def test_intervention_contract_rejects_invalid_mode_or_boundary(mode, boundary):
    with pytest.raises(ValueError):
        Circuit(fixture(), REGION, mode, boundary=boundary)


def test_clock_channel_and_external_shape_mismatches():
    with pytest.raises(ValueError):
        Circuit(fixture(), REGION, "replacement", boundary=Boundary(FakeEndpoint(), tuple(reversed(REGION)), 0.01))
    c = Circuit(fixture(), REGION)
    with pytest.raises(ValueError):
        c.advance((0,), 0.01)
    c = Circuit(fixture(), REGION, "replacement", boundary=Boundary(FakeEndpoint(), REGION, 0.01))
    with pytest.raises(ValueError):
        c.advance((0,) * 6, 0.02)


def test_lesion_clamps_every_selected_neuron_for_every_step():
    result = trial(Config(steps=120, seeds=(0,)), "lesion", 0)
    assert all(row["rates"][1:4] == [0.0] * 3 for row in result["trace"])
    assert all(row["command"] == 0.0 for row in result["trace"])


@pytest.mark.parametrize("seed", [0, 1, 2, 4294967295])
def test_native_sham_replacement_matches_intact_exactly(seed):
    config = Config(steps=150, seeds=(seed,), replacement="native")
    native = trial(config, "intact", seed)
    replacement = trial(config, "replacement", seed)
    assert native["trace"] == replacement["trace"]
    assert native["rmse"] == replacement["rmse"]


def test_trial_order_does_not_contaminate_rng_or_state_and_controls_are_paired():
    config = Config(steps=80, seeds=(0, 1))
    first = run(config)
    second = run(config, tuple(reversed(CONDITIONS)))
    key = lambda x: (x["seed"], x["condition"])
    assert sorted(first["trials"], key=key) == sorted(second["trials"], key=key)
    for seed in config.seeds:
        trials = [t for t in first["trials"] if t["seed"] == seed]
        for field in ("base_graph_sha256", "initial_state_sha256", "target_sha256", "exogenous_noise_sha256"):
            assert len({t[field] for t in trials}) == 1
    assert first == run(config)
    copy = dict(first)
    expected = copy.pop("result_sha256")
    assert digest(copy) == expected
    assert set(first["claims"].values()) == {False}


def test_sensory_feedback_is_not_frozen_to_baseline():
    config = Config(steps=150, seeds=(0,))
    a = trial(config, "intact", 0)
    b = trial(config, "lesion", 0)
    assert a["trace"][-1]["sense"] != b["trace"][-1]["sense"]
    assert a["target_sha256"] == b["target_sha256"]
    assert a["exogenous_noise_sha256"] == b["exogenous_noise_sha256"]


def test_recovery_is_null_when_lesion_does_not_degrade_this_trial():
    result = run(Config(steps=1, seeds=(0,)))
    assert result["paired_metrics"][0]["replacement_recovery_fraction"] is None
    assert result["paired_metrics"][0]["recovery_status"] == "lesion_not_worse_than_intact"


def test_exact_replay_reproduces_a_matched_native_trial():
    config = Config(steps=80, seeds=(7,), replacement="native")
    original = trial(config, "replacement", 7)
    replay = trial(dataclasses.replace(config, replacement="replay"), "replacement", 7,
                   lambda: ReplayEndpoint(original["replacement_recording"]))
    assert replay["trace"] == original["trace"]


def test_replay_rejects_changed_inputs_instead_of_faking_closed_loop():
    original = trial(Config(steps=30, seeds=(0,), replacement="native"), "replacement", 0)
    with pytest.raises(BoundaryError, match="request mismatch"):
        trial(Config(steps=30, seeds=(1,), replacement="replay"), "replacement", 1,
              lambda: ReplayEndpoint(original["replacement_recording"]))


@pytest.mark.parametrize("change", [
    lambda d: {**d, "schema": "other"}, lambda d: {**d, "units": "Hz"},
    lambda d: {**d, "evidence": "human_organoid"},
    lambda d: {**d, "frames_sha256": "invalid"}, lambda d: [],
    lambda d: {**d, "frames": []}, lambda d: {**d, "frames": "not-a-list"},
])
def test_bad_replay_metadata_and_checksum(change):
    b = Boundary(FakeEndpoint(), ("a",), 0.01)
    b.exchange(0, (0,))
    with pytest.raises(ValueError):
        ReplayEndpoint(change(replay_document(b.records)))


def test_replay_exhaustion_is_failure_and_recording_is_copied():
    b = Boundary(FakeEndpoint(), ("a",), 0.01)
    b.exchange(0, (0,))
    document = replay_document(b.records)
    replay = ReplayEndpoint(document)
    document["frames"].clear()
    consumer = Boundary(replay, ("a",), 0.01)
    assert consumer.exchange(0, (0,)) == (0.1,)
    with pytest.raises(BoundaryError, match="exhausted"):
        consumer.exchange(1, (0,))


def test_file_replay_and_cli_report(tmp_path):
    output = tmp_path / "result"
    assert main(["--out", str(output), "--steps", "25", "--seeds", "0", "--replacement", "native"]) == 0
    result = json.loads((output / "result.json").read_text())
    assert len(result["trials"]) == 4
    assert len(list(output.glob("*.csv"))) == 4
    assert "合成回路" in (output / "report.html").read_text()
    replay = output / "replacement-seed-0-replay.json"
    assert ReplayEndpoint.from_file(replay)
    replay_out = tmp_path / "replay"
    assert main(["--out", str(replay_out), "--steps", "25", "--seeds", "0",
                 "--replacement", "replay", "--replay", str(replay)]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(output)])
    assert exc.value.code == 2


@pytest.mark.parametrize("arguments", [
    ["--steps", "0"], ["--replacement", "replay"],
    ["--replay", "does-not-exist"], ["--dt", "nan"],
])
def test_cli_invalid_arguments_do_not_create_success_report(tmp_path, arguments):
    out = tmp_path / "result"
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(out), *arguments])
    assert exc.value.code == 2
    assert not out.exists()
