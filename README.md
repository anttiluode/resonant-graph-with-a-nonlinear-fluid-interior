# Resonant graph with a nonlinear fluid interior

A sparse resonant cavity graph wrapped around a nonlinear self-writing fluid interior.

The machine is built around one distinction:

```text
physical constraints  ->  operator  ->  waves occupying its modes
```

The graph does **not** store a dense learned matrix. Its 16 cavities are connected by only 22 undirected necks. Those sparse constraints define a stiffness/Laplacian operator, and at a drive frequency `ω` the global response is

```math
H(\omega)=\left[K-\omega^2I+i\gamma\omega I\right]^{-1}.
```

`H` is generally dense even though `K` is sparse.

Inside the graph is a 2-D incompressible Navier–Stokes field. Carrier-addressed A/B encounters are isolated with four exact-parity worlds and the collision-specific slow field is retained:

```math
\Delta\Omega_{AB}=P_{\rm slow}[\omega_{AB}-\omega_A-\omega_B+\omega_0].
```

Later cue A is injected back into that **distributed field itself**. There is no learned `Theta` matrix installed from the memory. The altered fluid changes the later cue trajectory at the B port, and that physical B-port response excites the outer resonant graph.

So the complete path is:

```text
sparse cavity constraints
        |
        v
coherent graph waves
        |
        v
addressed nonlinear fluid encounter
        |
        v
slow distributed fluid write
        |
        v
later cue travels differently
        |
        v
local cavity port
        |
        v
dense global resonant response
```

## The first receipt

The deterministic default run is in [`results/receipt.json`](results/receipt.json).

### Sparse world, dense effective operator

The default cavity geometry has only **22** undirected physical links out of 120 possible all-to-all links. At `ω = 4.10`, **98.44%** of entries of the Green's-function transfer matrix exceed 2% of its maximum magnitude.

The full transfer has effective rank **9.28**.

Then one local physical neck, cavity `0 <-> 1`, is strengthened by 25%. The resulting global transfer change has effective rank

```text
1.000000
```

while being dense across the graph.

That is not a fitted low-rank approximation. A single edge changes the graph Laplacian by an outer product, and the resolvent identity turns the corresponding global `ΔH` into a rank-one update.

This is a very direct bridge to the old `Kompressori` observation: **a local structural event can change a high-dimensional response operator through a low-rank global deformation.**

### Fewer sources than cavities

Six phased sources drive the sixteen-cavity graph. All 16 cavities respond above the same 2%-of-peak threshold. Two different six-phase source patterns produce different global resonant states (complex cosine similarity `0.650`).

The machine is therefore already doing the qualitative Berglund-like thing we wanted: a small number of phased sources address a larger family of global cavity responses through geometry rather than an explicit all-to-all matrix.

### The nonlinear interior writes a route

For a matched A+B carrier event, the retained collision-specific fluid field has

```text
slow-field norm                         1.75699e-2
fast / slow after washout               0.0895
```

After all training carriers are gone, cue A alone interacts with that stored fluid field and produces a signed response at the B port:

| memory condition | later B-port response |
| --- | ---: |
| matched phase 0 | **`-6.31486e-4`** |
| phase pi/2 | `-2.59460e-5` |
| phase pi | **`+5.90061e-4`** |
| frequency mismatch | `+6.33743e-7` |
| spatial separation | `-1.24081e-5` |

So matched recall is about **24.3×** quadrature, **996×** frequency mismatch and **50.9×** spatial separation. Antiphase reverses the sign of the later physical response.

### The learned fluid port wakes the whole graph

The matched B-port recall signal is then used as the physical excitation at cavity 4. The sparse resonant graph spreads it to **all 16 cavities** above the 2%-of-peak threshold. Its global response norm is `8.535e-4`.

For antiphase memory, the complete complex graph response is the opposite signed pattern (signed cosine `-1.0`) because the fluid changed the sign of the B-port event before the graph ever saw it.

No output MLP or dense learned attention matrix sits between the fluid memory and the global cavity response.

## What this means — and what it does not

The useful interpretation is:

```text
weights / parameters = physical constraints
operator             = propagation implied by those constraints
state                = waves currently occupying the operator's modes
learning             = a slow physical change that alters later propagation
```

The outer resonant graph is transformer-like only in a narrow sense: its sparse local constraints generate a dense global effective transfer, just as attention gives global token-to-token influence. It is **not** input-dependent softmax attention, and this repo does not claim transformer equivalence or superiority.

The nonlinear fluid interior supplies a state-dependent, history-dependent part that the bare resonant graph lacks. The current fluid geometry and graph/fluid ports are engineered. The machine does not discover semantic addresses, rewards, or useful cavity layouts autonomously.

## Why this repo exists

Several older lines now become parts of one object:

- **Berglund-style resonant cavities:** geometry determines the modes; fewer phased sources can excite a larger distributed pattern.
- **SighImageSuper:** the operator determines which modes persist and which decay.
- **InformationFlow / Wavebits:** space, carrier frequency and relative phase can address selected interactions.
- **SelectCarryBindWriteAskSelect:** nonlinear encounter -> changed medium -> later wave travels differently.
- **Kompressori:** local structural changes can create low-rank changes in a much larger response operator.
- **Mycelial Cortex:** a held field can be the recurrent state while sparse events provide the external interface; phase-sensitive bilinear channels preserve information that energy-only readouts lose.

The point here is not to repeat each old mechanism. It is to put the resonant **constraint -> mode -> nonlinear write -> new propagation** loop in one inspectable machine.

## Run

```bash
python -m pip install -e .[dev]
pytest -q
rgfi-run --out results/latest.json
```

Only NumPy is required by the machine itself.

See [`THEORY.md`](THEORY.md) and [`RESULTS.md`](RESULTS.md).

## Claim boundary

This is a numerical research toy.

Verified here: sparse cavity constraints produce a dense resonant Green's function; one local neck edit produces a dense rank-one transfer update; carrier-addressed Navier–Stokes interaction leaves a selective slow field; that field changes a later cue's physical B-port response; the resulting port event excites a global cavity pattern.

Not established: general learning, useful semantic routing, energy advantage, hardware feasibility at scale, transformer equivalence, brain equivalence, or any quantum advantage.
