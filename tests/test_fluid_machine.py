import numpy as np

from rgfi.fluid import FluidConfig
from rgfi.machine import ResonantFluidMachine


def test_fluid_memory_changes_later_cue_route_without_matrix_install():
    machine = ResonantFluidMachine.default(FluidConfig())
    matched = machine.learn_pair("A", "B", target_phase=0.0)
    anti = machine.learn_pair("A", "B", target_phase=np.pi)
    freq = machine.learn_pair("A", "B", target_omega=12.0)

    rm = machine.recall_pair(matched, "A", "B", steps=250)
    ra = machine.recall_pair(anti, "A", "B", steps=250)
    rf = machine.recall_pair(freq, "A", "B", steps=250)

    assert matched.fast_over_slow < 0.12
    assert abs(rm.target_signed_peak) > 5e-4
    assert rm.target_signed_peak * ra.target_signed_peak < 0.0
    assert abs(rm.target_signed_peak) > 100.0 * abs(rf.target_signed_peak)
    assert rm.target_over_distractor_abs > 20.0


def test_learned_fluid_port_drives_global_resonant_pattern():
    machine = ResonantFluidMachine.default(FluidConfig())
    matched = machine.learn_pair("A", "B", target_phase=0.0)
    out = machine.learned_recall_broadcast(matched, "A", "B")
    assert abs(out["fluid_recall"]["target_signed_peak"]) > 5e-4
    assert out["active_cavities_2pct"] == 16
    assert out["graph_pattern_l2"] > 1e-4
