from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .fluid import FluidConfig
from .machine import ResonantFluidMachine


def _signed_cosine(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.concatenate([a.real, a.imag])
    br = np.concatenate([b.real, b.imag])
    return float(np.dot(ar, br) / (np.linalg.norm(ar) * np.linalg.norm(br) + 1e-30))


def run_experiment(
    *,
    fluid_config: FluidConfig | None = None,
    recall_steps: int = 250,
    graph_omega: float = 4.10,
) -> dict:
    machine = ResonantFluidMachine.default(fluid_config)
    graph = asdict(machine.graph_diagnostics(graph_omega))

    memories = {
        "matched": machine.learn_pair("A", "B", target_phase=0.0),
        "quadrature": machine.learn_pair("A", "B", target_phase=np.pi / 2.0),
        "antiphase": machine.learn_pair("A", "B", target_phase=np.pi),
        "frequency_mismatch": machine.learn_pair("A", "B", target_omega=12.0),
        "spatial_separation": machine.learn_pair("A", "B", target_q=3.55),
    }

    broadcast = {
        name: machine.learned_recall_broadcast(
            memory,
            "A",
            "B",
            recall_steps=recall_steps,
            graph_omega=graph_omega,
        )
        for name, memory in memories.items()
    }

    fluid = {
        name: {
            "slow_norm": float(memory.slow_norm),
            "fast_over_slow_after_washout": float(memory.fast_over_slow),
            "target_signed_peak": float(broadcast[name]["fluid_recall"]["target_signed_peak"]),
            "target_peak_step": int(broadcast[name]["fluid_recall"]["target_peak_step"]),
            "distractor_signed_peak": float(broadcast[name]["fluid_recall"]["distractor_signed_peak"]),
            "target_over_distractor_abs": float(broadcast[name]["fluid_recall"]["target_over_distractor_abs"]),
        }
        for name, memory in memories.items()
    }

    matched_pattern = np.array(broadcast["matched"]["graph_pattern_real"]) + 1j * np.array(
        broadcast["matched"]["graph_pattern_imag"]
    )
    anti_pattern = np.array(broadcast["antiphase"]["graph_pattern_real"]) + 1j * np.array(
        broadcast["antiphase"]["graph_pattern_imag"]
    )

    matched = abs(fluid["matched"]["target_signed_peak"])
    quarter = abs(fluid["quadrature"]["target_signed_peak"])
    freq = abs(fluid["frequency_mismatch"]["target_signed_peak"])
    sep = abs(fluid["spatial_separation"]["target_signed_peak"])

    return {
        "status": "resonant_graph_with_nonlinear_fluid_interior",
        "claim_boundary": (
            "The outer cavity graph is a linear resonant network whose sparse physical "
            "constraints generate a dense Green's-function response. The interior write "
            "is a numerical 2-D incompressible Navier-Stokes collision residual used "
            "directly during later physical recall. The cavity/fluid ports and geometry "
            "are engineered; there is no learned dense attention matrix and no claim of "
            "quantum advantage or transformer superiority."
        ),
        "graph": graph,
        "fluid": fluid,
        "integrated_broadcast": {
            name: {
                "graph_pattern_l2": float(data["graph_pattern_l2"]),
                "active_cavities_2pct": int(data["active_cavities_2pct"]),
                "graph_pattern_abs": data["graph_pattern_abs"],
            }
            for name, data in broadcast.items()
        },
        "metrics": {
            "matched_over_quadrature_recall": float(matched / (quarter + 1e-15)),
            "matched_over_frequency_recall": float(matched / (freq + 1e-15)),
            "matched_over_spatial_recall": float(matched / (sep + 1e-15)),
            "antiphase_reverses_fluid_response": bool(
                fluid["matched"]["target_signed_peak"]
                * fluid["antiphase"]["target_signed_peak"]
                < 0.0
            ),
            "antiphase_graph_signed_cosine": _signed_cosine(
                matched_pattern, anti_pattern
            ),
            "matched_broadcast_active_cavities": int(
                broadcast["matched"]["active_cavities_2pct"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resonant graph + nonlinear fluid interior machine"
    )
    parser.add_argument("--fluid-grid", type=int, default=24)
    parser.add_argument("--fluid-train", type=int, default=210)
    parser.add_argument("--fluid-washout", type=int, default=700)
    parser.add_argument("--recall-steps", type=int, default=250)
    parser.add_argument("--graph-omega", type=float, default=4.10)
    parser.add_argument("--out", type=Path, default=Path("results/receipt.json"))
    args = parser.parse_args()
    config = FluidConfig(
        n=args.fluid_grid,
        train_steps=args.fluid_train,
        washout_steps=args.fluid_washout,
    )
    result = run_experiment(
        fluid_config=config,
        recall_steps=args.recall_steps,
        graph_omega=args.graph_omega,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
