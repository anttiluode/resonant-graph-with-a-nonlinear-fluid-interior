from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def effective_rank(matrix: Array) -> float:
    """Participation-ratio rank of the singular spectrum."""
    s = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    denom = float(np.sum(s * s))
    if denom <= 1e-30:
        return 0.0
    return float(np.sum(s) ** 2 / denom)


@dataclass(frozen=True)
class GraphReceipt:
    drive_frequency: float
    sparse_edges: int
    transfer_density_2pct: float
    full_transfer_effective_rank: float
    single_neck_delta_effective_rank: float
    single_neck_relative_delta_fro: float
    resonant_modal_effective_dimension: float
    dominant_mode_fraction: float
    six_source_pattern_cosine: float


class ResonantGraph:
    """Sparse physical constraints whose Green's function is dense.

    Think of each node as one resonant cavity and each non-zero edge as a neck,
    channel, spring or impedance link.  The *constraints* are sparse.  At a fixed
    frequency the response is the resolvent

        H(w) = [K - w^2 I + i gamma w I]^{-1},

    which is generally dense.  No dense learned matrix is stored.
    """

    def __init__(
        self,
        weights: Array,
        natural_frequency: Array,
        *,
        laplacian_scale: float = 2.2,
        damping: float = 0.08,
    ) -> None:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError("weights must be square")
        if not np.allclose(w, w.T):
            raise ValueError("cavity constraints must be symmetric")
        if np.any(w < 0.0):
            raise ValueError("neck weights must be non-negative")
        omega0 = np.asarray(natural_frequency, dtype=float)
        if omega0.shape != (w.shape[0],):
            raise ValueError("natural_frequency must have one value per cavity")
        self.weights = w.copy()
        self.natural_frequency = omega0.copy()
        self.laplacian_scale = float(laplacian_scale)
        self.damping = float(damping)
        self._rebuild()

    @classmethod
    def default(cls, n: int = 16) -> "ResonantGraph":
        if n != 16:
            raise ValueError("the default geometry currently has sixteen cavities")
        w = np.zeros((n, n), dtype=float)
        for i in range(n):
            j = (i + 1) % n
            w[i, j] = w[j, i] = 0.45
        # A few local physical shortcuts.  Still only 22 undirected links rather
        # than 120 possible all-to-all links.
        for i, j, value in (
            (0, 4, 0.18),
            (4, 8, 0.18),
            (8, 12, 0.18),
            (12, 0, 0.18),
            (2, 10, 0.12),
            (6, 14, 0.12),
        ):
            w[i, j] = w[j, i] = value
        idx = np.arange(n)
        omega0 = 4.0 + 0.12 * np.cos(2.0 * np.pi * idx / n)
        return cls(w, omega0)

    @property
    def n(self) -> int:
        return self.weights.shape[0]

    @property
    def edge_count(self) -> int:
        return int(np.count_nonzero(np.triu(self.weights, 1)))

    def _rebuild(self) -> None:
        lap = np.diag(self.weights.sum(axis=1)) - self.weights
        self.stiffness = (
            np.diag(self.natural_frequency**2)
            + self.laplacian_scale * lap
        )

    def copy(self) -> "ResonantGraph":
        return ResonantGraph(
            self.weights,
            self.natural_frequency,
            laplacian_scale=self.laplacian_scale,
            damping=self.damping,
        )

    def edit_neck(self, i: int, j: int, factor: float) -> None:
        """Physically change one local constraint; no global weights are edited."""
        i, j = int(i), int(j)
        if i == j or self.weights[i, j] <= 0.0:
            raise ValueError("edit_neck requires an existing off-diagonal neck")
        value = self.weights[i, j] * float(factor)
        if value < 0.0:
            raise ValueError("edited neck must remain non-negative")
        self.weights[i, j] = self.weights[j, i] = value
        self._rebuild()

    def transfer(self, omega: float) -> Array:
        eye = np.eye(self.n)
        op = (
            self.stiffness
            - float(omega) ** 2 * eye
            + 1j * self.damping * float(omega) * eye
        )
        return np.linalg.inv(op)

    def response(self, omega: float, source: Array) -> Array:
        source = np.asarray(source, dtype=complex)
        if source.shape != (self.n,):
            raise ValueError("source must have one complex amplitude per cavity")
        eye = np.eye(self.n)
        op = (
            self.stiffness
            - float(omega) ** 2 * eye
            + 1j * self.damping * float(omega) * eye
        )
        return np.linalg.solve(op, source)

    def modes(self) -> tuple[Array, Array]:
        values, vectors = np.linalg.eigh(self.stiffness)
        return np.sqrt(np.maximum(values, 0.0)), vectors

    def modal_response(self, omega: float, source: Array) -> Array:
        """Steady response coefficients in the cavity eigenmode basis."""
        values, vectors = np.linalg.eigh(self.stiffness)
        drive = vectors.T @ np.asarray(source, dtype=complex)
        denom = values - float(omega) ** 2 + 1j * self.damping * float(omega)
        return drive / denom

    def diagnostics(self, drive_frequency: float = 4.10) -> GraphReceipt:
        h0 = self.transfer(drive_frequency)
        dense_cut = 0.02 * float(np.max(np.abs(h0)))
        density = float(np.mean(np.abs(h0) > dense_cut))

        edited = self.copy()
        edited.edit_neck(0, 1, 1.25)
        h1 = edited.transfer(drive_frequency)
        dh = h1 - h0

        source = np.zeros(self.n, dtype=complex)
        source[0] = 1.0
        modal = self.modal_response(drive_frequency, source)
        power = np.abs(modal) ** 2
        modal_dim = float(power.sum() ** 2 / np.sum(power * power))
        dominant = float(np.max(power) / np.sum(power))

        six = np.array([0, 2, 5, 8, 11, 14])
        phase_a = np.array([0.0, 0.4, 1.1, 1.8, 2.4, 3.0])
        phase_b = np.array([0.0, 1.2, 2.4, 3.6, 4.8, 6.0])

        def pattern(phases: Array) -> Array:
            src = np.zeros(self.n, dtype=complex)
            src[six] = np.exp(1j * phases)
            return self.response(drive_frequency, src)

        p0, p1 = pattern(phase_a), pattern(phase_b)
        cosine = float(
            abs(np.vdot(p0, p1))
            / (np.linalg.norm(p0) * np.linalg.norm(p1) + 1e-30)
        )

        return GraphReceipt(
            drive_frequency=float(drive_frequency),
            sparse_edges=self.edge_count,
            transfer_density_2pct=density,
            full_transfer_effective_rank=effective_rank(h0),
            single_neck_delta_effective_rank=effective_rank(dh),
            single_neck_relative_delta_fro=float(
                np.linalg.norm(dh) / (np.linalg.norm(h0) + 1e-30)
            ),
            resonant_modal_effective_dimension=modal_dim,
            dominant_mode_fraction=dominant,
            six_source_pattern_cosine=cosine,
        )
