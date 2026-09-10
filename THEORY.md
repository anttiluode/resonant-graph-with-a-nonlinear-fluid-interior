# Theory: constraints are the weights, propagation is the operator

The machine separates three objects that ordinary neural-network language often collapses:

```text
constraints   C
the operator  H(C)
state         x(t)
```

The constraints are local and sparse. The operator they compile can be dense and global. The state is the live wave occupying the operator's modes.

## 1. Outer resonant graph

Let `W_ij` be the physical neck/coupling strength between cavities. The weighted graph Laplacian is

```math
L=D-W.
```

With local cavity resonance frequencies `omega_i`, define

```math
K=\operatorname{diag}(\omega_i^2)+\alpha L.
```

At drive frequency `omega`, the steady complex response to source vector `s` is

```math
x(\omega)=H(\omega)s,
```

```math
H(\omega)=\left[K-\omega^2I+i\gamma\omega I\right]^{-1}.
```

`K` is sparse because the physical world has only local channels. `H` is generally dense because a wave can reach a remote cavity through many paths.

This is the limited transformer analogy: global effective interaction emerges from local propagation rather than being stored as an explicit dense attention matrix.

## 2. Why one neck gives a low-rank global update

An undirected edge `(i,j)` with weight `w` contributes

```math
w(e_i-e_j)(e_i-e_j)^T
```

to the graph Laplacian. Changing only that neck by `delta w` therefore changes `K` by rank one:

```math
\Delta K
=\alpha\,\Delta w\,bb^T,
\qquad b=e_i-e_j.
```

For

```math
A=K-\omega^2I+i\gamma\omega I,
```

the Sherman–Morrison resolvent identity gives

```math
(A+ubb^T)^{-1}-A^{-1}
=
-\frac{u\,A^{-1}bb^TA^{-1}}
       {1+u\,b^TA^{-1}b}.
```

The right-hand side is an outer product: rank at most one.

So a **local physical edit can produce a dense but exactly low-rank change in the global operator**. That is not an analogy or a fitted numerical observation; it follows from the graph constraint algebra.

This is the cleanest bridge in this repo to the earlier Kompressori result.

## 3. Mode selection

The normal modes solve

```math
Kv_m=\lambda_m v_m,
qquad
\omega_m=\sqrt{\lambda_m}.
```

A source decomposes as

```math
s=\sum_m c_m v_m.
```

Its steady response is

```math
x(\omega)=
\sum_m
\frac{c_m}
{\lambda_m-\omega^2+i\gamma\omega}
v_m.
```

Driving near one resonance therefore selects only a few modes even though the response is spatially global. This is the Sigh-style principle in resonant form: **the operator determines which directions persist/amplify.**

## 4. Carrier addresses in the nonlinear interior

The interior is a periodic 2-D incompressible vorticity field:

```math
\partial_t\omega+u\cdot\nabla\omega
=\nu\Delta\omega+f,
```

```math
u=(\partial_y\psi,-\partial_x\psi),
\qquad -\Delta\psi=\omega.
```

A cavity port injects a localized oscillatory carrier. Its address is partly geometric and partly spectral:

```text
q          WHERE in the fluid
carrier k  WHICH spatial oscillation
omega      WHICH temporal channel
phase      signed relation to another carrier
```

The fluid is not handed the product of two carriers. The quadratic mixer is the Navier–Stokes convective term itself.

## 5. Collision-specific write

A changed final fluid state is not enough to claim an interaction memory. A alone and B alone each perturb the medium, and every world diffuses with time.

So the machine evolves four exact-parity worlds:

```text
W0   no carrier
WA   A only
WB   B only
WAB  A + B
```

After the carriers stop, all worlds receive the same forcing-free washout. The retained interaction memory is

```math
\Delta\Omega_{AB}
=P_{\rm slow}
[\omega_{AB}-\omega_A-\omega_B+\omega_0].
```

The subtraction occurs in the state itself before a nonlinear detector is applied.

## 6. Memory is the medium, not a matrix

Later recall initializes a new fluid world with the distributed slow field `DeltaOmega_AB` itself.

Cue A is then injected. To isolate only the interaction between the new cue and the stored physical field, three equal-time recall worlds are evolved:

```text
memory + A
memory only
A in blank fluid
```

and the measured cue-memory interaction is

```math
\chi_A(t)
=\omega_{\Delta\Omega+A}(t)
-\omega_{\Delta\Omega}(t)
-\omega_A(t).
```

A local B-cavity port measures `chi_A`. The stored field is never projected onto or installed into a learned `Theta` matrix.

This is the central statement of the repo:

> the memory does not represent the later routing operator; the memory **is part of the medium whose equations route the later signal**.

## 7. Graph / fluid seam

The graph and fluid need a physical seam. Here a named cavity owns a named local fluid port.

During learning, cavity-associated carriers enter the fluid at those ports. During recall, the signed local B-port response is the excitation returned to cavity B. The outer cavity graph then spreads that event through its own resonant Green's function.

This port coupling is engineered, just as the positions and necks of a laboratory resonator would be engineered. What is not engineered per memory is a dense routing matrix.

## 8. Relation to attention

A transformer computes an input-dependent matrix

```math
A(x)=\operatorname{softmax}(QK^T/\sqrt d)
```

and applies it to values.

This machine instead has

```math
H_C(\omega)
=\left[K(C)-\omega^2I+i\gamma\omega I\right]^{-1},
```

where `C` is physical structure plus the current fluid background.

The analogy is therefore only functional:

```text
attention matrix      <-> Green's-function global transfer
Q/K addressing        <-> position / frequency / phase coherence
values                 <-> carried wave amplitude/envelope
learned weights        <-> physical constraints / slow medium state
```

The important difference is that the field computer does not globally compare all pairs explicitly. It lets local propagation solve the global interaction problem.

Whether that ever produces a computational or energy advantage is an open empirical question.

## 9. The next deeper version

The current machine still treats the outer cavity graph and the inner fluid as two coupled substrates.

A stronger physical unification would linearize the later fast fluid dynamics around the learned slow background `Omega`:

```math
\partial_t\eta
=
-U(\Omega)\cdot\nabla\eta
-u(\eta)\cdot\nabla\Omega
+\nu\Delta\eta.
```

Then the learned fluid state directly changes the fast perturbation Jacobian `J_Omega`. Coupled to cavity coordinates, the whole device becomes one block operator whose Schur complement is the effective cavity-to-cavity Green's function.

That is the mathematically clean path to a true **resonant graph with a nonlinear fluid interior** in which the fluid background and cavity constraints jointly compile one operator.
