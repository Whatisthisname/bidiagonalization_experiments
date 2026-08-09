# Notes 01: Counting, redundancy, and multiplier gauge freedom

Everything here is elementary but is the root of the confusion, so it is
written out in full. Numerical verification: `exp01_counts_ranks_gauge.py`.
(If the notion of "gauge" itself is unfamiliar, read
`GUIDE_gauge_freedom.md` first -- a self-contained primer assuming only
linear algebra, with worked toy examples of everything used below.)

## 0. Convention (fixed once for the whole study)

For a constraint set `con(x, th) = 0` that (locally) determines the solver
output `x = s(th)`, and a loss `rho(x)`:

- **Adjoint system:** find mu with `(d con/d x)^T mu = -grad_x rho`.
- **Gradient:** `grad_th rho = (d con/d th)^T mu`.

Derivation: differentiate `con(s(th), th) = 0` to get
`(d con/dx)(ds/dth) = -(d con/dth)`; then
`grad_th rho = (ds/dth)^T grad_x rho = -(ds/dth)^T (d con/dx)^T mu
= (d con/dth)^T mu`.
This is the sign convention of the paper's section 4 (the boxed stable system
IS stationarity of `rho + <mu, con>`), *not* of its section 2.7, which flips
the sign of mu. One consistent convention is used in all notes and code here.

Note the two roles:

- The **stationarity system** has `dim(x) = d_U` *equations* (one per output
  coordinate) and `#rows(con)` *unknowns* (one multiplier per constraint row).
- So: more constraints than unknowns means an UNDERdetermined multiplier
  system, not an overdetermined one. This inversion is easy to trip over.

## 1. The unknowns

`x = (L, R, alpha, beta, res, c)` with `A` of size N x M and K iterations:

| block | size |
|---|---|
| L | NK |
| R | MK |
| alpha | K |
| beta | K-1 |
| res | M |
| c | 1 |
| **total d_U** | **NK + MK + M + 2K** |

## 2. The original constraint set (paper section 2.4, eqs (1)-(6))

| block | equation | rows |
|---|---|---|
| recR | A^T L = R B^T + res e_K^T | MK |
| recL | A R = L B | NK |
| init | R e_1 = c v | M |
| orthL | tril(L^T L) = I | K(K+1)/2 |
| orthR | tril(R^T R) = I | K(K+1)/2 |
| orthres | R^T res = 0 | K |
| **total d_O** | | **NK + MK + M + K^2 + 2K** |

So **d_O - d_U = K^2 exactly**. The constraint set is consistent (the exact
GKB output satisfies all rows) but redundant: rank(d con_full/d x) can be at
most d_U, and `exp01` verifies it equals d_U. Which rows are redundant, in
exact arithmetic, given the others:

- the strict lower triangles of orthL and orthR (2 * K(K-1)/2 = K^2 - K rows):
  orthogonality of the computed vectors follows from the recurrences plus unit
  norms by the classical GKB induction (spelled out in NOTES_02, sec. 4);
- orthres (K rows): res = A^T l_K - alpha_K r_K is automatically orthogonal to
  all r_n by the same induction.

Total redundancy K^2 - K + K = K^2, matching the count.

## 3. The reduced constraint set (paper appendix B.2)

| block | equation | rows |
|---|---|---|
| init | r_1 = c v | M |
| recl_n | alpha_n l_n = A r_n - beta_{n-1} l_{n-1} (beta_0 l_0 := 0) | NK |
| recr_n | beta_n r_{n+1} = A^T l_n - alpha_n r_n, n < K; res = A^T l_K - alpha_K r_K | MK |
| nrmL | l_n^T l_n = 1 | K |
| nrmR | r_n^T r_n = 1 | K |
| **total** | | **NK + MK + M + 2K = d_U** |

**Square.** And equivalent to the original set on the no-breakdown locus
(all alpha_n != 0, all beta_n != 0 for n < K): the reduced rows are a subset
(columns of the matrix equations, diagonals of the orthogonality blocks), and
conversely the reduced set implies full orthogonality by the GKB induction.
So both sets carve out the same solution manifold (a discrete union of
2^(2K)-ish sign choices; each branch is a graph over th).

## 4. Consequences for the adjoint

Let `J_red = d con_red/d x` (d_U x d_U) and `J_full = d con_full/d x`
(d_O x d_U). On the no-breakdown locus:

1. `J_red` is **nonsingular**. Proof sketch: order the unknowns in solve
   order (c, r_1), (alpha_1, l_1), (beta_1, r_2), ..., (res). In this order
   J_red is block lower triangular. The diagonal block of an l-step is the
   Jacobian of `(alpha, l) -> (alpha l - t, l^T l - 1)` which is
   `[[l, alpha I], [0, 2 l^T]]`, invertible iff alpha != 0 (explicit inverse
   in NOTES_02). Same for r-steps with beta_n, and the (c, r_1) block needs
   c != 0. So the reduced adjoint system
   `J_red^T mu = -grad_x rho`
   **has a unique solution for every loss**. It is solvable. NOTES_02 gives
   the explicit backward recurrence and exp02 verifies it against autodiff.

2. `J_full^T mu = -grad_x rho` is a consistent underdetermined system: its
   solution set is an affine subspace of dimension
   `d_O - rank(J_full) = d_O - d_U = K^2`.
   **Gauge freedom.** Any two solutions differ by an element of
   `null(J_full^T)`.

3. **Gradient invariance (the reason the gauge is harmless).** If
   `J_full^T mu_null = 0` then, differentiating `con_full(s(th), th) = 0`,
   `(d con_full/d th)^T mu_null = -(ds/dth)^T J_full^T mu_null = 0`.
   So every solution of the underdetermined system produces the SAME
   `grad_th`. Verified numerically in exp01 by sampling the null space.

4. The three objects to keep apart:
   - *the gradient*: unique, gauge-independent;
   - *the multipliers*: unique for con_red, a K^2-dimensional family for
     con_full;
   - *the algorithm that computes the multipliers*: a choice; different
     gauges admit different algorithms with different numerical behavior.

The paper's "stable adjoint system" will turn out to be: the con_full
stationarity system PLUS exactly K^2 gauge-fixing conditions (NOTES_03). The
appendix-B.1 "unstable adjoint" is the con_red system, i.e. the gauge in which
all K^2 extra multipliers are zero.

## 5. Verified numerically (exp01, float64, N=7, M=5, K=4 and N=6,M=6,K=3)

- counts d_U, d_O as above; rank(J_red) = d_U (invertible), rank(J_full) = d_U;
- dim null(J_full^T) = K^2;
- forward output satisfies both constraint sets to ~1e-14;
- least-squares multiplier for con_full and the unique multiplier for con_red
  give identical gradients, matching plain autodiff to ~1e-13;
- adding random null(J_full^T) elements to the multiplier leaves the gradient
  unchanged to ~1e-13.
