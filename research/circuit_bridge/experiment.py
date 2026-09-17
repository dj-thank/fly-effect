"""Paired experiments against a synthetic 1-D plant, not a fly body."""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from .core import (SCHEMA, UNITS, UPSTREAM, Boundary, Circuit, Edge, Endpoint,
                   Graph, NativeEndpoint, SurrogateEndpoint, digest, integer,
                   positive, replay_document)

CONDITIONS = ("intact", "lesion", "replacement", "shuffled")
REGION = ("c0", "c1", "c2")


def fixture() -> Graph:
    return Graph(("sense", *REGION, "motor_plus", "motor_minus"), (
        Edge("sense", "c0", 1.2), Edge("c0", "c1", 0.9),
        Edge("c1", "c2", 0.7), Edge("c2", "c0", -0.5),
        Edge("c0", "c2", 0.25), Edge("c1", "c0", 0.35),
        Edge("c2", "c1", -0.3), Edge("c1", "motor_plus", 1.0),
        Edge("c1", "motor_minus", -1.0)))


@dataclass(frozen=True)
class Config:
    dt_s: float = 0.01
    steps: int = 400
    seeds: tuple[int, ...] = (0, 1, 2)
    tau_s: float = 0.04
    mass: float = 1.0
    damping: float = 1.2
    gain: float = 2.0
    replacement: str = "surrogate"

    def __post_init__(self) -> None:
        for name in ("dt_s", "tau_s", "mass", "damping", "gain"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        if self.dt_s > 0.02 or self.dt_s * self.damping / self.mass > 0.5:
            raise ValueError("Plant integration timestep exceeds fixture stability budget")
        integer(self.steps, "steps", 1, 100000)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not 1 <= len(self.seeds) <= 32 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be nonempty, unique, and at most 32")
        for seed in self.seeds:
            integer(seed, "seed", 0, 2**32 - 1)
        if self.steps * len(self.seeds) * 4 > 1000000:
            raise ValueError("Experiment exceeds one million condition-steps")
        if self.replacement not in {"surrogate", "native", "replay"}:
            raise ValueError("Unknown replacement endpoint")


@dataclass(frozen=True)
class Plant:
    position: float = 0.0
    velocity: float = 0.0

    def advance(self, command: float, config: Config) -> Plant:
        velocity = self.velocity + config.dt_s * (
            config.gain * command - config.damping * self.velocity) / config.mass
        position = self.position + config.dt_s * velocity
        if not math.isfinite(position) or not math.isfinite(velocity):
            raise ValueError("Nonfinite plant state")
        return Plant(position, velocity)


def trial(config: Config, condition: str, seed: int,
          endpoint_factory: Callable[[], Endpoint] | None = None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    integer(seed, "seed", 0, 2**32 - 1)
    graph = fixture()
    effective = graph.shuffled(REGION, seed ^ 0xA57C) if condition == "shuffled" else graph
    boundary = None
    if condition == "replacement":
        if endpoint_factory is not None:
            endpoint = endpoint_factory()
        elif config.replacement == "native":
            endpoint = NativeEndpoint(graph, REGION, config.tau_s)
        elif config.replacement == "surrogate":
            endpoint = SurrogateEndpoint(len(REGION), config.tau_s)
        else:
            raise ValueError("Replay requires a fresh endpoint factory for each trial")
        boundary = Boundary(endpoint, REGION, config.dt_s)
    mode = "intact" if condition == "shuffled" else condition
    circuit = Circuit(effective, REGION, mode, config.tau_s, boundary)
    plant = Plant()
    initial_sha = digest({"circuit": circuit.state, "plant": asdict(plant)})
    rng = random.Random(seed)
    noise = [rng.gauss(0.0, 0.002) for _ in range(config.steps)]
    targets = [0.15 + 0.4 * math.sin(step * config.dt_s * 1.2)
               for step in range(config.steps)]
    rows = []
    for step, (noise_value, target) in enumerate(zip(noise, targets)):
        before = plant
        sense = math.tanh(target - before.position + noise_value)
        state = circuit.advance((sense, 0, 0, 0, 0, 0), config.dt_s)
        command = (state[-2] - state[-1]) / 2.0
        plant = plant.advance(command, config)
        rows.append({"step": step, "t_s": step * config.dt_s,
                     "end_t_s": (step + 1) * config.dt_s,
                     "target": target, "noise": noise_value, "sense": sense,
                     "position_before": before.position, "velocity_before": before.velocity,
                     "position_after": plant.position, "velocity_after": plant.velocity,
                     "command": command, "rates": list(state)})
    rmse = math.sqrt(sum((r["target"] - r["position_before"])**2 for r in rows) / len(rows))
    return {"condition": condition, "seed": seed, "base_graph_sha256": graph.fingerprint,
            "effective_graph_sha256": effective.fingerprint,
            "shuffle_is_noop": condition == "shuffled" and effective == graph,
            "initial_state_sha256": initial_sha, "target_sha256": digest(targets),
            "exogenous_noise_sha256": digest(noise), "region": list(REGION),
            "rmse": rmse, "trace_sha256": digest(rows), "trace": rows,
            "replacement_recording": replay_document(boundary.records) if boundary else None}


def run(config: Config, condition_order: Sequence[str] = CONDITIONS,
        endpoint_factory: Callable[[], Endpoint] | None = None) -> dict:
    if len(condition_order) != 4 or set(condition_order) != set(CONDITIONS):
        raise ValueError("All four conditions are required exactly once")
    trials = [trial(config, condition, seed, endpoint_factory)
              for seed in config.seeds for condition in condition_order]
    paired = []
    for seed in config.seeds:
        group = {t["condition"]: t for t in trials if t["seed"] == seed}
        for name in ("initial_state_sha256", "target_sha256", "exogenous_noise_sha256"):
            if len({item[name] for item in group.values()}) != 1:
                raise AssertionError(f"Unmatched experimental condition: {name}")
        loss = group["lesion"]["rmse"] - group["intact"]["rmse"]
        valid = loss > 1e-12
        # Descriptive fixture metric, never clamp it into an apparent success.
        recovery = ((group["lesion"]["rmse"] - group["replacement"]["rmse"]) / loss
                    if valid else None)
        paired.append({"seed": seed, "lesion_rmse_increase": loss,
                       "replacement_recovery_fraction": recovery,
                       "recovery_status": "descriptive_only" if valid else "lesion_not_worse_than_intact"})
    result = {"schema": SCHEMA, "upstream_base_commit": UPSTREAM,
              "evidence": "synthetic_fixture", "units": UNITS,
              "body": "synthetic_1d_damped_plant_not_fly_body", "config": asdict(config),
              "claims": {"real_connectome_loaded": False, "upstream_brain_executed": False,
                         "fly_body_executed": False, "living_tissue_connected": False,
                         "autonomous_fly_behavior_demonstrated": False},
              "interpretation": "Software validation only; no biological efficacy or learning claim.",
              "trials": trials, "paired_metrics": paired}
    result["result_sha256"] = digest(result)
    return result
