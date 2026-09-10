from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class CarrierPort:
    """One graph cavity coupled to one addressed carrier port in the fluid."""

    name: str
    cavity: int
    q: float
    omega: float = 4.0
    phase: float = 0.0


@dataclass(frozen=True)
class FluidConfig:
    n: int = 24
    length: float = 2.0 * np.pi
    dt: float = 0.015
    viscosity: float = 0.04
    k_split: float = 3.0
    carrier_k: float = 6.0
    packet_sigma: float = 0.55
    packet_amplitude: float = 8.0
    train_steps: int = 210
    washout_steps: int = 700


@dataclass
class CollisionMemory:
    slow: Array
    slow_norm: float
    fast_over_slow: float


@dataclass(frozen=True)
class RecallReceipt:
    target_signed_peak: float
    target_peak_step: int
    distractor_signed_peak: float
    distractor_peak_step: int
    target_over_distractor_abs: float


class NavierStokes2D:
    r"""Small pseudo-spectral incompressible 2-D Navier--Stokes solver.

    Vorticity form:

        d_t omega + u dot grad omega = nu Laplacian omega + f
        u = (d_y psi, -d_x psi),  -Laplacian psi = omega.

    The nonlinear term is de-aliased with the 2/3 rule.  Diffusion is handled by
    an exponential step and the nonlinear/forcing term by a two-stage ETD-RK
    update.  The code is intentionally tiny rather than a general CFD package.
    """

    def __init__(self, config: FluidConfig) -> None:
        self.config = config
        self.n = int(config.n)
        self.length = float(config.length)
        self.dt = float(config.dt)
        self.nu = float(config.viscosity)

        k = np.fft.fftfreq(self.n, d=self.length / self.n) * 2.0 * np.pi
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx * self.kx + self.ky * self.ky
        self.k2_inv = np.zeros_like(self.k2)
        mask = self.k2 > 0.0
        self.k2_inv[mask] = 1.0 / self.k2[mask]

        kmax = float(np.max(np.abs(k))) * (2.0 / 3.0)
        self.dealias = (np.abs(self.kx) <= kmax) & (np.abs(self.ky) <= kmax)
        self.slow_mask = np.sqrt(self.k2) <= float(config.k_split)

        axis = np.linspace(0.0, self.length, self.n, endpoint=False)
        self.x, self.y = np.meshgrid(axis, axis, indexing="ij")
        self.omega = np.zeros((self.n, self.n), dtype=float)

        lop = -self.nu * self.k2
        self._E = np.exp(lop * self.dt)
        self._phi1 = np.ones_like(lop)
        nonzero = np.abs(lop * self.dt) > 1e-10
        self._phi1[nonzero] = (
            np.expm1(lop[nonzero] * self.dt) / (lop[nonzero] * self.dt)
        )

    def _nonlinear_hat(self, omega_hat: Array) -> Array:
        psi_hat = omega_hat * self.k2_inv
        u = np.fft.ifft2(1j * self.ky * psi_hat).real
        v = np.fft.ifft2(-1j * self.kx * psi_hat).real
        omega_x = np.fft.ifft2(1j * self.kx * omega_hat).real
        omega_y = np.fft.ifft2(1j * self.ky * omega_hat).real
        return -np.fft.fft2(u * omega_x + v * omega_y) * self.dealias

    def step(self, forcing: Array | None = None) -> None:
        omega_hat = np.fft.fft2(self.omega)
        forcing_hat = (
            0.0
            if forcing is None
            else np.fft.fft2(np.asarray(forcing, dtype=float)) * self.dealias
        )
        n1 = self._nonlinear_hat(omega_hat) + forcing_hat
        a_hat = self._E * omega_hat + self.dt * self._phi1 * n1
        n2 = self._nonlinear_hat(a_hat) + forcing_hat
        omega_hat = a_hat + self.dt * self._phi1 * (n2 - n1)
        self.omega = np.fft.ifft2(omega_hat).real

    def low_pass(self, field: Array) -> Array:
        return np.fft.ifft2(np.fft.fft2(field) * self.slow_mask).real


class NonlinearFluidInterior:
    """Carrier-addressed nonlinear write and later direct physical recall.

    A and B are trained in four equal-time worlds.  The stored state is the
    collision-specific slow field

        P_slow[omega_AB - omega_A - omega_B + omega_0].

    Later recall does *not* project that field onto a learned matrix.  It puts the
    distributed field back into the Navier--Stokes medium, injects A, and measures
    how A travels differently because that physical field exists.
    """

    def __init__(self, config: FluidConfig | None = None) -> None:
        self.config = config or FluidConfig()

    def _periodic_delta(self, x: Array, center: float) -> Array:
        d = x - center
        return (d + 0.5 * self.config.length) % self.config.length - 0.5 * self.config.length

    def packet(
        self,
        sim: NavierStokes2D,
        port: CarrierPort,
        time: float,
        *,
        amplitude: float | None = None,
    ) -> Array:
        x0 = 0.5 * self.config.length
        y0 = float(port.q) % self.config.length
        dx = self._periodic_delta(sim.x, x0)
        dy = self._periodic_delta(sim.y, y0)
        sigma = float(self.config.packet_sigma)
        envelope = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        angle = float(port.q) % (2.0 * np.pi)
        spatial = float(self.config.carrier_k) * (
            np.cos(angle) * dx + np.sin(angle) * dy
        )
        temporal = np.cos(float(port.omega) * time + float(port.phase))
        amp = self.config.packet_amplitude if amplitude is None else float(amplitude)
        return amp * temporal * envelope * np.cos(spatial)

    def collision_memory(
        self,
        source: CarrierPort,
        target: CarrierPort,
    ) -> CollisionMemory:
        worlds = {
            name: NavierStokes2D(self.config)
            for name in ("W0", "WA", "WB", "WAB")
        }
        for step in range(int(self.config.train_steps)):
            t = step * self.config.dt
            a = self.packet(worlds["W0"], source, t)
            b = self.packet(worlds["W0"], target, t)
            worlds["W0"].step()
            worlds["WA"].step(a)
            worlds["WB"].step(b)
            worlds["WAB"].step(a + b)

        for _ in range(int(self.config.washout_steps)):
            for world in worlds.values():
                world.step()

        total = (
            worlds["WAB"].omega
            - worlds["WA"].omega
            - worlds["WB"].omega
            + worlds["W0"].omega
        )
        slow = worlds["W0"].low_pass(total)
        fast = total - slow
        slow_norm = float(np.linalg.norm(slow))
        return CollisionMemory(
            slow=slow,
            slow_norm=slow_norm,
            fast_over_slow=float(
                np.linalg.norm(fast) / (slow_norm + 1e-15)
            ),
        )

    def _probe(self, sim: NavierStokes2D, q: float, sigma: float) -> Array:
        dx = self._periodic_delta(sim.x, 0.5 * self.config.length)
        dy = self._periodic_delta(sim.y, float(q))
        mask = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        return mask / max(float(np.linalg.norm(mask)), 1e-15)

    @staticmethod
    def _signed_peak(values: list[float]) -> tuple[float, int]:
        data = np.asarray(values, dtype=float)
        idx = int(np.argmax(np.abs(data)))
        return float(data[idx]), idx

    def recall(
        self,
        memory: CollisionMemory | Array,
        source: CarrierPort,
        target: CarrierPort,
        *,
        steps: int = 250,
        cue_amplitude: float = 3.0,
        probe_sigma: float = 0.45,
        distractor_q: float = 4.55,
    ) -> RecallReceipt:
        slow = memory.slow if isinstance(memory, CollisionMemory) else np.asarray(memory, dtype=float)
        cue_memory = NavierStokes2D(self.config)
        memory_only = NavierStokes2D(self.config)
        cue_blank = NavierStokes2D(self.config)
        cue_memory.omega = np.array(slow, copy=True)
        memory_only.omega = np.array(slow, copy=True)

        cue = self.packet(
            cue_memory,
            replace(source, phase=0.0),
            0.0,
            amplitude=cue_amplitude,
        )
        cue_memory.omega += cue
        cue_blank.omega += cue

        target_probe = self._probe(cue_memory, target.q, probe_sigma)
        distractor_probe = self._probe(cue_memory, distractor_q, probe_sigma)
        target_values: list[float] = []
        distractor_values: list[float] = []
        for _ in range(int(steps)):
            cue_memory.step()
            memory_only.step()
            cue_blank.step()
            # Only the interaction between the stored field and the new cue.
            cross = cue_memory.omega - memory_only.omega - cue_blank.omega
            target_values.append(float(np.sum(target_probe * cross)))
            distractor_values.append(float(np.sum(distractor_probe * cross)))

        target_peak, target_step = self._signed_peak(target_values)
        distractor_peak, distractor_step = self._signed_peak(distractor_values)
        return RecallReceipt(
            target_signed_peak=target_peak,
            target_peak_step=target_step,
            distractor_signed_peak=distractor_peak,
            distractor_peak_step=distractor_step,
            target_over_distractor_abs=float(
                abs(target_peak) / (abs(distractor_peak) + 1e-15)
            ),
        )


def default_ports() -> dict[str, CarrierPort]:
    """Four named graph cavities coupled to addressed points in the fluid."""
    return {
        "A": CarrierPort("A", cavity=0, q=1.45, omega=4.0),
        "B": CarrierPort("B", cavity=4, q=2.15, omega=4.0),
        "C": CarrierPort("C", cavity=8, q=4.55, omega=7.0),
        "D": CarrierPort("D", cavity=12, q=5.15, omega=9.5),
    }
