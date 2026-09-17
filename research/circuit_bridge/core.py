"""Immutable sparse circuits and a clocked, fail-closed replacement boundary.

All values are dimensionless rate-model states, NOT measured firing rates or
stimulation commands. This module never loads or modifies the upstream CNS.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

SCHEMA = "fly-effect-circuit-bridge/v1"
UNITS = "dimensionless-rate-state"
UPSTREAM = "89d59fc56c3e4f3b0f9a26725ce4ee55fdf1ee36"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def positive(value: object, name: str) -> float:
    result = finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def vector(values: Iterable[float], size: int, name: str,
           bound: float | None = None) -> tuple[float, ...]:
    result = tuple(finite(v, name) for v in values)
    if len(result) != size:
        raise ValueError(f"{name}: expected {size} values, got {len(result)}")
    if bound is not None and any(abs(v) > bound for v in result):
        raise ValueError(f"{name}: outside [-{bound}, {bound}]")
    return result


@dataclass(frozen=True)
class Edge:
    pre: str
    post: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", finite(self.weight, "edge weight"))


@dataclass(frozen=True)
class Graph:
    ids: tuple[str, ...]
    edges: tuple[Edge, ...]
    evidence: str = "synthetic_fixture"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", tuple(self.ids))
        object.__setattr__(self, "edges", tuple(self.edges))
        if not 1 <= len(self.ids) <= 10000 or len(self.edges) > 100000:
            raise ValueError("Graph exceeds the small-circuit resource budget")
        if any(not isinstance(n, str) or not n for n in self.ids):
            raise ValueError("Neuron IDs must be nonempty strings")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate neuron IDs")
        known = set(self.ids)
        if any(not isinstance(e, Edge) or e.pre not in known or e.post not in known
               for e in self.edges):
            raise ValueError("Edge references unknown neurons")
        if self.evidence != "synthetic_fixture":
            raise ValueError("This v1 runner accepts synthetic fixtures only")

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    def region(self, ids: Sequence[str]) -> tuple[str, ...]:
        ids = tuple(ids)
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(self.ids):
            raise ValueError("Region must contain unique, known neuron IDs")
        return ids

    def shuffled(self, region: Sequence[str], seed: int) -> Graph:
        """Permute only internal destinations; keep all boundary edges unchanged.

        Edge multiplicity, source/weight pairs and destination counts are kept.
        Parallel edges and self-loops are permitted. Some seeds are no-ops; the
        experiment manifest explicitly reports that degeneracy.
        """
        selected = set(self.region(region))
        integer(seed, "seed", 0, 2**32 - 1)
        indices = [i for i, e in enumerate(self.edges)
                   if e.pre in selected and e.post in selected]
        posts = [self.edges[i].post for i in indices]
        random.Random(seed).shuffle(posts)
        edges = list(self.edges)
        for i, post in zip(indices, posts):
            edges[i] = Edge(edges[i].pre, post, edges[i].weight)
        return Graph(self.ids, tuple(edges), self.evidence)


@dataclass(frozen=True)
class Request:
    step: int
    dt_s: float
    channel_ids: tuple[str, ...]
    drive: tuple[float, ...]
    units: str = UNITS

    def __post_init__(self) -> None:
        integer(self.step, "step", 0, 1000000)
        object.__setattr__(self, "dt_s", positive(self.dt_s, "dt_s"))
        ids = tuple(self.channel_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(n, str) or not n for n in ids):
            raise ValueError("Invalid channel IDs")
        object.__setattr__(self, "channel_ids", ids)
        object.__setattr__(self, "drive", vector(self.drive, len(ids), "drive"))
        if self.units != UNITS:
            raise ValueError("Unit mismatch")

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class Response:
    step: int
    dt_s: float
    channel_ids: tuple[str, ...]
    values: tuple[float, ...]
    request_sha256: str
    units: str = UNITS

    @classmethod
    def for_request(cls, request: Request, values: Sequence[float]) -> Response:
        return cls(request.step, request.dt_s, request.channel_ids,
                   tuple(values), request.fingerprint)


class Endpoint(Protocol):
    def exchange(self, request: Request) -> Response: ...


class BoundaryError(RuntimeError):
    """No stale, invalid or missing values may advance the circuit/body."""


class Boundary:
    def __init__(self, endpoint: Endpoint, channel_ids: Sequence[str], dt_s: float,
                 deadline_s: float = 1.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.endpoint = endpoint
        self.channel_ids = tuple(channel_ids)
        self.dt_s = positive(dt_s, "dt_s")
        self.deadline_s = positive(deadline_s, "deadline_s")
        # Reuse Request validation for the ordered port schema.
        Request(0, self.dt_s, self.channel_ids, (0.0,) * len(self.channel_ids))
        self.clock = clock
        self.next_step = 0
        self.failed = False
        self.records: list[dict] = []

    def exchange(self, step: int, drive: Sequence[float]) -> tuple[float, ...]:
        if self.failed:
            raise BoundaryError("Boundary is latched after failure; create a fresh trial")
        try:
            if type(step) is not int or step != self.next_step:
                raise ValueError("Nonconsecutive step")
            request = Request(step, self.dt_s, self.channel_ids, tuple(drive))
            start = self.clock()
            response = self.endpoint.exchange(request)
            elapsed = self.clock() - start
            if not math.isfinite(elapsed) or elapsed < 0 or elapsed > self.deadline_s:
                raise ValueError("Endpoint missed its soft deadline")
            if not isinstance(response, Response):
                raise ValueError("Endpoint returned no Response")
            if (type(response.step) is not int or response.step != request.step
                    or finite(response.dt_s, "response dt") != request.dt_s
                    or tuple(response.channel_ids) != self.channel_ids
                    or response.units != UNITS
                    or response.request_sha256 != request.fingerprint):
                raise ValueError("Response clock, channels, units or request hash mismatch")
            values = vector(response.values, len(self.channel_ids), "response", 1.0)
            self.records.append({"request": asdict(request), "response": asdict(response)})
            self.next_step += 1
            return values
        except Exception as exc:
            self.failed = True
            raise BoundaryError(f"Boundary stopped at step {step}: {exc}") from exc


class NativeEndpoint:
    """Identity/sham replacement, to verify the boundary itself changes nothing."""
    def __init__(self, graph: Graph, region: Sequence[str], tau_s: float) -> None:
        self.ids = graph.region(region)
        self.index = {n: i for i, n in enumerate(self.ids)}
        self.edges = [(self.index[e.pre], self.index[e.post], e.weight)
                      for e in graph.edges if e.pre in self.index and e.post in self.index]
        self.tau_s = positive(tau_s, "tau_s")
        self.state = [0.0] * len(self.ids)

    def exchange(self, request: Request) -> Response:
        if request.channel_ids != self.ids:
            raise ValueError("Native endpoint port mismatch")
        drive = list(request.drive)
        for pre, post, weight in self.edges:
            drive[post] += weight * self.state[pre]
        alpha = -math.expm1(-request.dt_s / self.tau_s)
        self.state = [old + alpha * (math.tanh(d) - old)
                      for old, d in zip(self.state, drive)]
        return Response.for_request(request, self.state)


class SurrogateEndpoint:
    """Deliberately different, untrained synthetic dynamics; not an organoid."""
    def __init__(self, size: int, tau_s: float) -> None:
        integer(size, "size", 1, 10000)
        self.state = [0.0] * size
        self.tau_s = positive(tau_s, "tau_s")

    def exchange(self, request: Request) -> Response:
        drive = vector(request.drive, len(self.state), "surrogate drive")
        alpha = -math.expm1(-request.dt_s / self.tau_s)
        previous = self.state
        self.state = [old + alpha * (math.tanh(
            0.6 * d + 0.45 * previous[(i - 1) % len(previous)]
            - 0.1 * previous[(i + 1) % len(previous)]) - old)
            for i, (old, d) in enumerate(zip(previous, drive))]
        return Response.for_request(request, self.state)


class ReplayEndpoint:
    """Exact input-conditioned file replay, NOT responsive living tissue.

    A different input/clock/port fails closed; no future frame look-ahead,
    interpolation, repetition of stale frames, or silently ignoring inputs.
    """
    def __init__(self, document: dict) -> None:
        if not isinstance(document, dict):
            raise ValueError("Replay document must be an object")
        if (document.get("schema") != SCHEMA or document.get("units") != UNITS
                or document.get("evidence") != "synthetic_fixture"):
            raise ValueError("Replay must declare v1 synthetic dimensionless provenance")
        if not isinstance(document.get("frames"), list):
            raise ValueError("Replay frames must be a list")
        self.frames = json.loads(canonical(document["frames"]))
        if not self.frames or len(self.frames) > 100000:
            raise ValueError("Replay frame count exceeds budget")
        if document.get("frames_sha256") != digest(self.frames):
            raise ValueError("Replay checksum mismatch")
        self.cursor = 0

    @classmethod
    def from_file(cls, path: Path) -> ReplayEndpoint:
        with Path(path).open("rb") as stream:
            raw = stream.read(20_000_001)
        if len(raw) > 20_000_000:
            raise ValueError("Replay exceeds 20 MB budget")
        return cls(json.loads(raw))

    def exchange(self, request: Request) -> Response:
        if self.cursor >= len(self.frames):
            raise ValueError("Replay exhausted")
        frame = self.frames[self.cursor]
        if digest(frame["request"]) != request.fingerprint:
            raise ValueError("Replay request mismatch; this is not a live adapter")
        result = dict(frame["response"])
        result["channel_ids"] = tuple(result["channel_ids"])
        result["values"] = tuple(result["values"])
        response = Response(**result)
        self.cursor += 1
        return response


def replay_document(records: list[dict]) -> dict:
    return {"schema": SCHEMA, "units": UNITS, "evidence": "synthetic_fixture",
            "frames": records, "frames_sha256": digest(records)}


class Circuit:
    def __init__(self, graph: Graph, region: Sequence[str], mode: str = "intact",
                 tau_s: float = 0.04, boundary: Boundary | None = None) -> None:
        if mode not in {"intact", "lesion", "replacement"}:
            raise ValueError("Unknown circuit intervention")
        self.graph = graph
        self.region = graph.region(region)
        self.index = {n: i for i, n in enumerate(graph.ids)}
        self.selected = {self.index[n] for n in self.region}
        self.edges = tuple((self.index[e.pre], self.index[e.post], e.weight)
                           for e in graph.edges)
        self.tau_s = positive(tau_s, "tau_s")
        self.mode = mode
        self.boundary = boundary
        if (mode == "replacement") != (boundary is not None):
            raise ValueError("Exactly replacement mode requires a boundary")
        if boundary is not None and boundary.channel_ids != self.region:
            raise ValueError("Circuit/boundary channel order mismatch")
        self.state = (0.0,) * len(graph.ids)
        self.step = 0

    def advance(self, external: Sequence[float], dt_s: float) -> tuple[float, ...]:
        dt_s = positive(dt_s, "dt_s")
        if self.boundary is not None and dt_s != self.boundary.dt_s:
            raise ValueError("Circuit/boundary timestep mismatch")
        drive = list(vector(external, len(self.state), "external drive"))
        # Assemble outside inputs first, then the target's internal inputs. Both
        # native/sham paths use the same summation order for exact replay tests.
        for pre, post, weight in self.edges:
            if pre in self.selected and post in self.selected:
                continue
            drive[post] += weight * self.state[pre]
        target_drive = tuple(drive[self.index[n]] for n in self.region)
        if self.mode == "intact":
            for pre, post, weight in self.edges:
                if pre in self.selected and post in self.selected:
                    drive[post] += weight * self.state[pre]
        alpha = -math.expm1(-dt_s / self.tau_s)
        state = [old + alpha * (math.tanh(d) - old)
                 for old, d in zip(self.state, drive)]
        if self.mode == "lesion":
            for index in self.selected:
                state[index] = 0.0
        if self.boundary is not None:
            values = self.boundary.exchange(self.step, target_drive)
            for n, value in zip(self.region, values):
                state[self.index[n]] = value
        # Commit only after a valid boundary response.
        self.state = vector(state, len(state), "circuit state", 1.0)
        self.step += 1
        return self.state
