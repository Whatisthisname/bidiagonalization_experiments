# Notes 03: The stable adjoint system IS the adjoint of the original constraints, gauge-fixed

Numerical verification: `exp03_stable_multipliers.py`, `exp04_gauge_fixed_dense_solve.py`.

## 1. Direct differentiation of the ORIGINAL constraints

Take the Lagrangian of the original set (NOTES_01 sec. 2) with multipliers
Xi (recR), Phi (recL), kappa (init), Sigma lower-tri (orthL), Omega lower-tri
(orthR), eta (orthres):

    Lag = rho + <Xi, A^T L - R B^T - res e_K^T> + <Phi, A R - L B>
              + <kappa, R e_1 - c v>
              + <Sigma, tril(L^T L - I)> + <Omega, tril(R^T R - I)>
              + <eta, R^T res>

Stationarity, block by block (grad_x rho written dL, dR, da, db, dres, dc):

    d/dL:    0 = dL + A Xi - Phi B^T + L (Sigma + Sigma^T)            [NK]
    d/dR:    0 = dR + A^T Phi - Xi B + R (Omega + Omega^T)
                    + kappa e_1^T + res eta^T                         [MK]
    d/dres:  0 = dres - Xi e_K + R eta                                [M]
    d/dc:    0 = dc - v^T kappa                                       [1]
    d/da_n:  0 = da_n - (L^T Phi)_{nn} - (R^T Xi)_{nn}                [K]
    d/db_n:  0 = db_n - (L^T Phi)_{n,n+1} - (R^T Xi)_{n+1,n}          [K-1]

and the gradient is grad_A = Phi R^T + L Xi^T, grad_v = -c kappa.

**Compare with the paper's boxed "stable adjoint system for
bidiagonalization": the d/dL, d/dR, d/dres, d/dc equations above ARE,
verbatim, the boxed equations.** No Hessenberg objects were harmed in this
derivation. What the boxed system has *in addition* are the "reprojection
constraints"

    I_>= o (L^T Phi)   = superdiag(db)   i.e.  l_m^T phi_n = db_{n-1}[m=n-1], m <= n
    I_>=+1 o (R^T Xi)  = diag(da)        i.e.  r_m^T xi_n  = da_n[m=n],     m <= n+1

which have K^2 + 2K - 1 entries. Of these, exactly 2K - 1 linear
combinations reproduce the d/da_n and d/db_n stationarity rows above
(diagonal sums and cross sums). The remaining

    (K^2 + 2K - 1) - (2K - 1) = K^2

conditions are NOT stationarity conditions of anything: they are **gauge
fixing** -- they pick one point of the K^2-dimensional affine solution family
identified in NOTES_01.

## 2. Verified numerically (exp03)

At the computed forward solution, with the paper's stable backward recurrence
reimplemented transparently and its multipliers recorded (float64):

- the multipliers satisfy the con_full stationarity system to ~1e-15 (C1);
- they satisfy all K^2 + 2K - 1 selection identities to ~5e-16 (C2);
- the reduced-gauge multiplier (exp02) also satisfies stationarity (~1e-14)
  but violates the selection identities by O(10)-O(100) at K=4 already;
- the difference of the two multiplier vectors is a null vector of
  J_full^T (residual ~1e-13) and the two gradients agree to ~1e-15 (C3).

## 3. The square characterization (exp04) -- the "shortcut" that replaces the detour

Define the linear system in mu = (Xi, Phi, kappa, Sigma, Omega, eta):

    (a) stationarity rows for all x-coordinates EXCEPT alpha, beta;
    (b) all K^2 + 2K - 1 selection rows.

Row count: (d_U - 2K + 1) + (K^2 + 2K - 1) = d_U + K^2 = d_O = #unknowns.

exp04 verifies: this square matrix is **nonsingular** (cond ~1e2 at the
tested sizes), its unique solution **equals the stable recurrence's
multipliers to ~1e-15**, the dropped alpha/beta rows hold automatically, and
the resulting gradient matches autodiff.

Consequences:

1. **The stable adjoint system can be both stated and derived without ever
   introducing the augmented Hessenberg problem.** Recipe: write the
   Lagrangian of the original constraints (1)-(6); observe the multiplier
   family is K^2-dimensional; impose the selection "all computable inner
   products of the multipliers against the computed bases are pinned to the
   loss-supplied values"; check squareness by counting; solve. Solving this
   square system by back-substitution in the natural column order
   *reproduces the paper's backward recurrence line by line* (the Sigma-row
   formula is the Schur complement of the selection rows in the d/dL rows).
2. Nonsingularity of this square system on the no-breakdown locus is proved
   DIRECTLY (no Hessenberg) by induction in NOTES_05; the back-substitution
   of that proof is the paper's backward recurrence itself. What the
   Hessenberg detour remains: an alternative proof/derivation and the
   discovery route by which the specific selection was found. (The boxed
   system's eta and xi_K assignment equations are the step-0 output of the
   induction: they are the [d/dres rows + column-K selection rows] solved
   for (eta, xi_K).)

## 4. Why this particular gauge is the numerically good one

The selection pins **every inner product of the multiplier columns with the
already-computed Krylov vectors that the recurrence will ever touch**
(rows m <= n resp. m <= n+1 -- exactly the "past" at step n of the backward
sweep). These are the components in which the forward basis actually lives;
in the reduced gauge they are exactly the components that grow
exponentially (the exact multiplier's in-span part reaches 1e10 while the
gradient stays O(1e2); exp13 -- the reduced recurrence computes them
RELATIVELY accurately, but their sheer size caps the assembled gradient at
eps |mu|/|grad|; NOTES_02 sec. 3, NOTES_06 sec. 2). In the stable gauge
those components are not *computed* at all -- they are *prescribed* (known
in closed form from the cotangents da, db, hence O(|cotangent|) by fiat),
and mid-flight reprojection re-imposes them on the drifting carries. Gauge
freedom is what makes prescribing them consistent.

Note the pinned values are loss-dependent (they involve da, db). So the
gauge choice is not one fixed subspace of multiplier space; it is an affine
selection depending linearly on the cotangent. This matters for NOTES_04
(what a constraint set could or could not produce).

## 5. Where the Hessenberg K^2 went (bookkeeping of the "paradox")

The augmented Hessenberg system (on \"A-umlaut\" of size (N+M) x (N+M), 2K
steps) is exactly square: d_O^hess = d_U^hess; its adjoint has a UNIQUE
multiplier. Restricting to the alternating sparsity pattern:

| augmented object | rows | trivialized by sparsity | surviving rows |
|---|---|---|---|
| lower-tri orthogonality of Q (K(2K+1)) | mixed-parity pairs | K^2 (become 0=0) | K(K+1) = orthL + orthR |
| recurrence columns (2K(N+M)) | wrong-parity halves | K(N+M) | KN + KM = recL + recR |
| init column (N+M) | N-half | N | M = init |
| Q^T r residual orth (2K) | even rows | K | K = orthres |

The K^2 trivialized orthogonality rows are precisely the mixed-parity pairs
q_i^T q_j = 0 whose entries vanish IDENTICALLY under the sparsity ansatz;
their multipliers are the checkerboard zeros of Gamma in the paper's
Theorem 4.1. In the reduced (bidiagonal) variable space those rows simply do
not exist, and their K^2 multipliers are gone -- yet the surviving
constraints still overdetermine the smaller variable space by K^2. In other
words:

- augmented world: square constraints, unique multiplier, no gauge;
- bidiagonal world: K^2 redundant constraints, K^2-dim multiplier gauge;
- the pruning map sends the unique augmented multiplier to ONE point of the
  bidiagonal gauge family -- and exp03/exp04 show that point is exactly the
  gauge fixed by the selection identities. (Later verified DIRECTLY: exp16
  builds the dense augmented adjoint, prunes it, and recovers Phi, Xi,
  kappa, eta of the stable gauge to 3e-15. The checkerboard zeros
  themselves have a half-page parity proof, NOTES_07.)

So the "conceptual confusion" dissolves: dimension counts of CONSTRAINTS
control uniqueness of MULTIPLIERS, not of gradients; equivalent constraint
sets share the gradient but not the multiplier; the augmentation is a
device that turns a redundant description into a square one by adding
variables (the structural zeros) that make the redundant rows do real work.
