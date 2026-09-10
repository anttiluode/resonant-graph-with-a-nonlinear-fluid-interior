from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from .fluid import (
    CarrierPort,
    CollisionMemory,
    FluidConfig,
    NonlinearFluidInterior,
    RecallReceipt,
    default_ports,
)
from .graph import GraphReceipt, ResonantGraph

Array = np.ndarray


class ResonantFluidMachine:
    """Sparse resonant graph around a nonlinear self-writing fluid interior.

    The outer graph provides stable identity and a sparse set of physical
    constraints.  Its frequency-domain Green's function is dense.  The inner
    fluid supplies carrier-addressed nonlinear binding and a distributed slow
    state that changes how a later cue travels.

    There is no learned dense matrix.  The only persistent objects are graph
    constraints and the distributed fluid memory itself.
    """

    def __init__(
        self,
        graph: ResonantGraph | None = None,
        fluid: NonlinearFluidInterior | None = None,
        ports: dict[str, CarrierPort] | None = None,
    ) -> None:
        self.graph = graph or ResonantGraph.default()
        self.fluid = fluid or NonlinearFluidInterior()
        self.ports = dict(ports or default_ports())
        for port in self.ports.values():
            if not 0 <= port.cavity < self.graph.n:
                raise ValueError(f"port {port.name} points outside the graph")

    @classmethod
    def default(cls, fluid_config: FluidConfig | None = None) -> "ResonantFluidMachine":
        return cls(
            graph=ResonantGraph.default(),
            fluid=NonlinearFluidInterior(fluid_config),
            ports=default_ports(),
        )

    def learn_pair(
        self,
        source_name: str,
        target_name: str,
        *,
        target_phase: float | None = None,
        target_omega: float | None = None,
        target_q: float | None = None,
    ) -> CollisionMemory:
        source = self.ports[source_name]
        target = self.ports[target_name]
        changes = {}
        if target_phase is not None:
            changes["phase"] = float(target_phase)
        if target_omega is not None:
            changes["omega"] = float(target_omega)
        if target_q is not None:
            changes["q"] = float(target_q)
        if changes:
            target = replace(target, **changes)
        return self.fluid.collision_memory(source, target)

    def recall_pair(
        self,
        memory: CollisionMemory | Array,
        source_name: str,
        target_name: str,
        *,
        steps: int = 250,
    ) -> RecallReceipt:
        return self.fluid.recall(
            memory,
            self.ports[source_name],
            self.ports[target_name],
            steps=steps,
        )

    def cavity_broadcast(
        self,
        cavity: int,
        amplitude: complex,
        *,
        omega: float = 4.10,
    ) -> Array:
        source = np.zeros(self.graph.n, dtype=complex)
        source[int(cavity)] = complex(amplitude)
        return self.graph.response(omega, source)

    def learned_recall_broadcast(
        self,
        memory: CollisionMemory | Array,
        source_name: str,
        target_name: str,
        *,
        recall_steps: int = 250,
        graph_omega: float = 4.10,
    ) -> dict:
        """Ask the learned fluid route, then let that physical port excite the graph.

        The scalar at the interface is simply the signed local detector amplitude
        of the target cavity's physical neck.  It is not a learned projection of
        the distributed memory.  The graph then propagates that port excitation
        through its own sparse physical constraints.
        """
        recall = self.recall_pair(
            memory, source_name, target_name, steps=recall_steps
        )
        target = self.ports[target_name]
        pattern = self.cavity_broadcast(
            target.cavity, recall.target_signed_peak, omega=graph_omega
        )
        mag = np.abs(pattern)
        active = int(np.count_nonzero(mag > 0.02 * max(float(mag.max()), 1e-30)))
        return {
            "fluid_recall": asdict(recall),
            "graph_drive_cavity": int(target.cavity),
            "graph_drive_frequency": float(graph_omega),
            "graph_pattern_real": pattern.real.tolist(),
            "graph_pattern_imag": pattern.imag.tolist(),
            "graph_pattern_abs": mag.tolist(),
            "graph_pattern_l2": float(np.linalg.norm(pattern)),
            "active_cavities_2pct": active,
        }

    def six_source_pattern(
        self,
        phases: Array,
        *,
        omega: float = 4.10,
    ) -> Array:
        indices = np.array([0, 2, 5, 8, 11, 14])
        phases = np.asarray(phases, dtype=float)
        if phases.shape != (6,):
            raise ValueError("six source phases required")
        source = np.zeros(self.graph.n, dtype=complex)
        source[indices] = np.exp(1j * phases)
        return self.graph.response(omega, source)

    def graph_diagnostics(self, omega: float = 4.10) -> GraphReceipt:
        return self.graph.diagnostics(omega)
