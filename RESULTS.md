# Results ledger

Deterministic default receipt: [`results/receipt.json`](results/receipt.json).

## Graph

At drive frequency `4.10`:

```text
cavities                              16
undirected physical necks             22 / 120 possible
Green's-function entries >2% max      98.4375%
full transfer effective rank           9.2773
```

A 25% edit to only neck `0 <-> 1` changes the complete transfer matrix by relative Frobenius norm `0.01927`.

The effective rank of that dense global `Delta H` is

```text
1.000000
```

as expected from the rank-one Laplacian edge update and the resolvent identity.

A single cavity drive near resonance has modal participation ratio `3.185`; the strongest eigenmode carries `48.88%` of steady modal power.

Six phased sources at cavities `[0,2,5,8,11,14]` excite all 16 cavities above 2% of the maximum response. Two different six-source phase codes produce global complex response cosine `0.6503`.

## Nonlinear fluid memory

A matched phase-0 A+B training event leaves

```text
collision-specific slow norm          1.75699e-2
fast / slow after 700-step washout    0.08951
```

Later A-only recall measures the interaction between the new cue and the stored fluid field, subtracting both memory-only evolution and A in blank fluid.

| stored condition | B signed peak | B / distant distractor |
| --- | ---: | ---: |
| matched phase 0 | **-6.31486e-4** | 45.55x |
| phase pi/2 | -2.59460e-5 | 20.78x |
| phase pi | **+5.90061e-4** | 42.16x |
| frequency mismatch | +6.33743e-7 | 8.79x |
| spatial separation | -1.24081e-5 | 88.84x |

Matched / control B-response ratios:

```text
matched / quadrature          24.34x
matched / frequency mismatch 996.44x
matched / spatial separation  50.89x
```

Antiphase reverses the later physical B response.

## Integrated graph + fluid path

The B-port response from fluid recall is inserted at cavity B (`cavity 4`) as the local physical graph excitation. No learned dense matrix is inserted between them.

For matched memory:

```text
B-port input amplitude         -6.31486e-4
global cavity response L2       8.53534e-4
cavities above 2% of peak      16 / 16
```

The antiphase memory produces the opposite complex global response pattern (signed cosine `-1.0`) because the sign reversal occurs inside the fluid recall before resonant broadcast.

## What actually fell out

The most general result in this first build is not the particular fluid number. It is the operator algebra of the cavity graph:

> a single local constraint edit produces a dense but rank-one change in the global Green's-function transfer.

That gives a concrete physical interpretation of the earlier low-rank `Delta J` observations: locality of structural change need not imply locality of computational consequence.

The fluid result adds the missing history dependence: a carrier encounter can leave a distributed physical state that changes a later cue before the graph globally broadcasts it.

## Claim boundary

These are deterministic numerical toy results, not hardware measurements.

No benchmark against transformers is claimed. No semantic task is solved. Port geometry, cavity layout, drive frequency, spectral split and detector locations are chosen by us. The useful result is that the full causal path is executable without a learned dense matrix:

```text
local constraints -> global resonant operator
carrier collision -> physical slow memory
physical memory + later cue -> local port event
graph resonance -> global response
```
