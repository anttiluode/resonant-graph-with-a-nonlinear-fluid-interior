import numpy as np

from rgfi.graph import ResonantGraph, effective_rank


def test_sparse_constraints_generate_dense_green_function():
    graph = ResonantGraph.default()
    h = graph.transfer(4.10)
    assert graph.edge_count == 22
    assert np.mean(np.abs(h) > 0.02 * np.max(np.abs(h))) > 0.90
    assert effective_rank(h) > 5.0


def test_single_neck_edit_is_global_but_low_rank():
    graph = ResonantGraph.default()
    h0 = graph.transfer(4.10)
    edited = graph.copy()
    edited.edit_neck(0, 1, 1.25)
    dh = edited.transfer(4.10) - h0
    # One symmetric edge modifies the Laplacian by an outer product.  The
    # resolvent identity therefore produces a rank-one dense response update.
    assert effective_rank(dh) < 1.01
    assert np.mean(np.abs(dh) > 0.001 * np.max(np.abs(dh))) > 0.90
    assert np.linalg.norm(dh) / np.linalg.norm(h0) > 0.01


def test_six_sources_control_sixteen_cavity_pattern():
    graph = ResonantGraph.default()
    indices = np.array([0, 2, 5, 8, 11, 14])
    src = np.zeros(16, dtype=complex)
    src[indices] = np.exp(1j * np.array([0.0, 0.4, 1.1, 1.8, 2.4, 3.0]))
    response = graph.response(4.10, src)
    assert response.shape == (16,)
    assert np.count_nonzero(np.abs(response) > 0.02 * np.max(np.abs(response))) == 16
