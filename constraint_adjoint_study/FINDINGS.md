# Findings: constraint sets, gauges, and the stable adjoint of bidiagonalization

Session 2026-07-28, 21:26-22:50 (16 experiments, 7 notes; all re-run end to
end at 22:27, all pass). All numbers below reproduce with
`python exp0X_*.py` in this folder (float64 verifications at machine
precision unless noted). Notation and conventions: NOTES_01.

## 1. The two premises, verified

- **d_O > d_U, by exactly K^2.** The original constraint set (paper eqs
  (1)-(6)) has d_O = NK + MK + M + K^2 + 2K rows for d_U = NK + MK + M + 2K
  unknowns; rank(d con_full/dx) = d_U, so the redundancy is exactly K^2
  = strict-lower orthogonality rows (K^2 - K) + R^T res rows (K), which the
  GKB induction (NOTES_02 sec. 4) derives from the rest. [exp01]
- **The B.2 reduced set is equivalent and square** (d_U rows), and its
  Jacobian d con_red/dx is nonsingular on the no-breakdown locus (explicit
  block-triangular inverse). [NOTES_01-02, exp01]

## 2. The conceptual knot, untied

"More constraints than unknowns" affects the MULTIPLIERS, not the gradient:

- Differentiating a constraint set gives d_U stationarity equations in
  #rows-many multipliers. For con_full that is UNDERdetermined: a
  K^2-dimensional affine family of multipliers, every member of which gives
  the SAME gradient (invariance lemma + numerical check to 1e-15). [exp01]
- For con_red the multiplier is unique -- and the system IS solvable, by an
  explicit backward recurrence (NOTES_02 sec. 2), verified to reproduce
  autodiff gradients to 1e-14. [exp02]
- **Where "I cannot solve it" came from:** two documented traps. (a) If the
  appendix-B.3 matrix form is read with TRIANGULAR Sigma, Omega (natural
  after section 4), the system is genuinely underdetermined -- null space of
  dimension exactly K^2 - K, verified. The B.2-derived system has DIAGONAL
  Sigma, Omega; then it is square and nonsingular. (b) The vector equation
  at step n must be paired with the scalar rows (S-al_n)/(S-be_n) to
  eliminate sigma_n before dividing by alpha_n; attempted in any other
  order, the step looks circular. [exp02 C4; NOTES_02 sec. 2]

## 3. What the stable system actually is

**The paper's boxed stable adjoint system = the stationarity system of the
ORIGINAL constraints (1)-(6), plus exactly K^2 gauge-fixing rows.**

- The boxed d/dL, d/dR, d/dres, d/dc equations coincide verbatim with the
  Lagrangian stationarity of (1)-(6); the stable recurrence's multipliers
  satisfy them to 1e-15. [exp03]
- The "reprojection constraints" of the boxed system contain the remaining
  stationarity rows (d/dalpha, d/dbeta) as diagonal/cross sums; the other
  K^2 rows are pure gauge choice ("pin every inner product of the
  multipliers against the computed basis to loss-supplied values").
- The square linear system [stationarity minus alpha/beta rows + selection
  rows], assembled from the original constraints ONLY, is nonsingular
  (cond ~1e2 at small sizes) and its unique solution equals the stable
  recurrence's multipliers to 1.4e-15. [exp04]
- Direct induction proof of nonsingularity/consistency, no Hessenberg
  anywhere; the back-substitution IS the paper's recurrence, and enforcing
  the selection rows on the carries IS reprojection. [NOTES_05]
- The reduced gauge (all extra multipliers zero) and the stable gauge differ
  by an O(100) null-space vector already at K = 4, with identical gradients.
  [exp03]
- Bookkeeping of the Hessenberg detour: in the lifted space the K^2
  redundant rows become the mixed-parity orthogonality constraints that pin
  the structural-zero blocks; the lifted system is square, its unique
  multiplier prunes exactly to the stable gauge (checkerboard zeros of the
  paper's Theorem = multipliers of the trivialized rows). [NOTES_03 sec. 5]

## 4. Why gauges matter numerically (squareness is worthless, gauge is everything)

Float32 end-to-end, normal ensemble n x n/2, K = n/2, random all-output
cotangent, error vs validated float64 reference [exp05]:

| n | autodiff32 | reduced32 | reduced64 (!) | stable32 no-reproj | stable32 reproj |
|---|---|---|---|---|---|
| 16 | NaN | 8.4e-05 | 2.0e-13 | 1.9e-06 | 7.8e-07 |
| 32 | NaN | 1.6e+02 | 3.2e-07 | 7.0e-04 | 1.6e-05 |
| 48 | NaN | 3.1e+06 | 3.9e-03 | 9.6e-03 | 1.3e-04 |
| 64 | NaN | 2.0e+10 | 2.7e+01 | 2.0e+02 | 3.0e-05 |
| 96 | NaN | 4.4e+20 | 8.5e+11 | 6.7e+05 | 6.8e-06 |
| 128 | NaN | NaN | 7.1e+23 | 4.4e+09 | 3.4e-04 |

- The reduced gauge -- the UNIQUE multiplier of a perfectly square,
  perfectly nonsingular system -- diverges exponentially even in FLOAT64:
  ~0.59 decimal digits lost per iteration (fit over K = 8..48, three seeds;
  same exponent in float32; the 1/|beta| division proxy is negligible).
  [exp10]
- **The mechanism (exp13, 60-digit arithmetic): the loss is NOT solver
  error.** The reduced recurrence computes every multiplier column to
  ~2e-16 RELATIVE accuracy; what grows exponentially is the exact
  multiplier itself (0.68 digits per backward step at (32,16,16):
  |lam_1| ~ 7e10 against |grad_A| ~ 4e2). The gradient then dies in the
  assembly by cancellation, with a provable floor
  (eps_arith + eps_primal) * |mu|_inf/|grad| that matches the observed
  errors in float64 and float32 (even the exact 60-digit multiplier yields
  a 2.3e-7 gradient error, because the float64 primal's 1e-15 orthogonality
  drift is amplified by the same factor). **Stability of an adjoint
  formulation = smallness of its multiplier representation**; exp10's rate
  is the growth of |mu(K)|, not accumulated roundoff. [exp13, NOTES_06
  sec. 2]
- The stable gauge without mid-flight reprojection eventually blows up too;
  with reprojection it is flat at the primal-limited floor. Reprojection =
  re-imposing the gauge rows on the carries; it exists BECAUSE the gauge
  rows are known in advance -- which is possible only in the redundant
  formulation. [exp05, NOTES_05 remark 2]

## 5. The search for a "direct" constraint set (the conjectured great find)

Two rigorous negative results and one positive surprise:

- **Exact reproduction is impossible for square sets in the original
  variables.** (a) res-block obstruction: any square recurrence-shaped set
  forces (-rho_K) = dres, while the stable gauge has xi_K = dres + R eta;
  the gap equals R eta exactly (verified to 1e-15, |R eta| = O(1), at full
  and partial depth). (b) Span obstruction: reproducing the stable
  (Phi, Xi, kappa) for all cotangents requires representing a rank-(K^2+2K)
  "leftover" map within the <= 2K-dimensional range available to the extra
  rows of any square completion; rank verified = K^2 + 2K with clean
  spectral gap. [exp07, exp08(a), exp09; NOTES_04 sec. A]
- **Behavioral reproduction is possible -- and practical.** The square
  Gram-Schmidt set V3 (projectors applied to the classic residuals + unit
  norms) has: identical on-manifold solutions, a unique multiplier DIFFERENT
  from the stable one, and an adjoint whose conditioning and float32
  dense-solve accuracy track the gauge-fixed stable system essentially
  digit-for-digit across sizes (16-96), seeds, ensembles (normal,
  Hilbert-type), and depths (K = M, K = M/2) -- while con_red's adjoint
  conditioning explodes (1.8e5 -> 1.5e19 by n = 48). Projecting the raw
  vectors instead of the residuals (V1/V2) is markedly worse. Mechanism:
  differentiating the projectors manufactures the same L-, R-supported
  triangular correction terms that the stable gauge gets from the
  orthogonality multipliers -- a square projector set is
  "self-gauge-fixing". [exp06, exp08, exp09; NOTES_04 sec. B]
- **The V3 adjoint solves by an O(K) backward recurrence of the same cost
  as the paper's method** (two matvecs per step; derivation in NOTES_04
  sec. B.1): the projectors arise from differentiation, the paper's
  reprojection targets (da, db) arise as the alpha/beta stationarity pins
  l_j^T lam_j = -da_j, r_{j+1}^T rho_j = -db_j, and carried multipliers
  enter only in projected form ("reprojection with implicit targets").
  Verified: float64 gradient matches autodiff to 3e-16 and the multiplier
  matches the dense solve of the full V3 Jacobian to 2e-15; in float32 it
  matches the paper's stable recurrence to 2-3 digits at EVERY tested
  configuration up to n = 128, K = 64 (normal + Hilbert, full + half depth).
  [exp11]
- **Carry-correspondence theorem (the sharp form of "behavioral"):** at the
  exact primal, Phi_n + lam_n \in span(l_1..l_n) and Xi_n + rho_n \in
  span(r_1..r_{n+1}) -- the paper's carries and the V3 carries agree
  IDENTICALLY in all future-band, residual-direction, and out-of-span
  components, and differ only in the past band (which the stable gauge
  overwrites with pinned values and V3 annihilates with projectors). Proof
  in three lines by sandwiching the shared gradient between frame vectors,
  using the selection identities on one side and the projector structure on
  the other; verified to 1e-15 at three configurations (future band agrees
  to 5e-16 while past band differs O(1)). The two methods are one algorithm
  in two gauges of a band that neither propagates. The reduced carry, by
  contrast, differs O(1)-O(10) in the future band as well. [exp12, NOTES_06
  sec. 1]
- **Floating-point corollary:** with a native float32 primal the two
  methods' mutual disagreement is 3x-1000x smaller than their (common)
  error -- the error budget is dominated by the primal imperfection pushed
  through the theorem-identical propagation; with a cast-down float64
  primal their private rounding is independent and equal-sized
  (mutual/err ~ 1). Both are primal-limited; improving the gradient
  further means improving the forward pass. [exp14, NOTES_06 sec. 1.1]

## 6. Implications for the paper

1. **Not invalidated -- reframed.** The stable system does not need the
   augmented Hessenberg problem to be stated, derived, proved solvable, or
   solved (NOTES_03 + NOTES_05 give the two-page direct route: Lagrangian of
   (1)-(6), K^2 gauge freedom, selection rows, induction). The detour
   remains a valid alternative derivation and the discovery route, and the
   sparsity theorem acquires a nice interpretation (multipliers of
   trivialized constraints vanish). Recommend adding the direct
   derivation/proof as a remark or appendix; it makes the contribution
   self-contained and explains WHY reprojection is legitimate (gauge rows,
   known in advance, enforceable at will).
1b. **The sparsity theorem's induction appendix can be replaced by a
   half-page parity argument** (NOTES_07 + exp16): the checkerboard and all
   alternation patterns are the fixed-point equations of a Z2 symmetry of
   the augmented adjoint system; the same argument proves the primal Q/H/r
   patterns and explains why the zeros are exact in floating point. If the
   detour is kept in the paper, this is a major simplification of its
   longest proof.
2. **Appendix B.3 as typeset is underdetermined** if Sigma, Omega are read
   as triangular (null dimension K^2 - K, exp02): state explicitly that the
   unstable system's Sigma, Omega are DIAGONAL, or write l_n sigma_n,
   r_n omega_n columnwise as in B.1. Also worth stating in B.1/B.3 that this
   system, though uniquely solvable in exact arithmetic (backward recurrence
   in NOTES_02), has an exponentially large multiplier (~0.6 digits/step),
   which caps any finite-precision gradient at eps * |mu|/|grad| -- that is
   the precise content of "unstable", and it is a property of the
   formulation, not of any particular solver (exp13). Finally, the
   appendix intro's "equivalent to the
   stable system" should be qualified: equivalent at the GRADIENT level
   only; the multipliers differ by an O(100) gauge vector already at K = 4
   (exp03), and the appendix system has K^2 fewer multiplier dof. (Minor: in
   B.1/B.3 the running index switches between n, i and k, K; and B.3's
   "delta_{i,1} kappa" should read "kappa e_1^T" in matrix form.)
3. **The appendix-B.2 equivalence claim** ("one can show by induction") can
   cite the explicit induction (NOTES_02 sec. 4), including the no-breakdown
   hypotheses alpha_n != 0, beta_n != 0, c != 0 it needs.
4. **The V3 alternative deserves a mention (or a follow-up note):** a
   square Gram-Schmidt constraint set whose plain Lagrangian adjoint --
   unique multiplier, no gauge analysis, no augmentation, no explicit
   reprojection rules -- gives an equally stable O(K) recurrence (exp11),
   provably carrying the same information as the paper's recurrence outside
   the pinned band (exp12, NOTES_06 sec. 1) and sharing its error budget in
   float32 (exp14). It cannot reproduce the paper's exact system (sec. 5),
   so the paper's contribution stands, but "the detour is required for
   stability" is not the right framing; "the detour is one route to one
   (canonical) member of a family of stable formulations" is.

## 7. Open threads

- ~~WHY ~0.6 digits/step of multiplier growth~~ RESOLVED by exp15: the rate
  is the backward loop gain, rate <= mean_n log10(|A|_2^2/(alpha_n beta_n))
  -- tight for the Hilbert-type ensemble (predicted 4.8-4.9 vs measured
  4.85-5.10 digits/step; the exact multiplier reaches 2.6e70 at (32,16,16),
  mpmath-confirmed), an upper bound with ~0.3 digits/step slack for the
  normal ensemble (0.9 vs 0.6). It is NOT the forward drift rate of
  un-reorthogonalized GKB (comparable for normal, 14x off for Hilbert:
  forward drift saturates, the adjoint gain does not). What remains open is
  only the ensemble-statistics question of the slack for well-conditioned
  problems.
- ~~A symmetry proof of the checkerboard sparsity theorem~~ DONE
  (NOTES_07, exp16): S = diag(I_N, -I_M) anticommutes with the augmented
  matrix; the adjoint system is equivariant under (Lambda, Gamma, lambda_0,
  gamma) -> (-S Lambda D, D Gamma D, -S lambda_0, -D gamma) with
  D = diag((-1)^i); uniqueness forces the solution to be a fixed point,
  which IS the full Theorem 4.1 (checkerboard Gamma, alternating gamma and
  Lambda blocks, M-block lambda_0). Half a page, replaces the paper's
  induction appendix, and also proves the primal Q/H/r patterns. Verified:
  all claimed zeros are exactly 0.0 in the dense solve; the equivariance
  residual is 1e-15; and the pruned augmented multiplier equals the stable
  gauge (Phi, Xi, kappa, eta) to 3e-15 -- the last link of the pruning
  chain, previously only argued structurally.
- ~~Floating-point version of the carry correspondence~~ RESOLVED by exp14:
  the float32 agreement between the two methods is due to the dominant
  error being common-mode (primal imperfection through identical
  propagation), while their private rounding is independent and equal-sized
  (visible once the primal is cast down from float64). No first-order
  "same rounding errors" claim is needed, and none holds.
