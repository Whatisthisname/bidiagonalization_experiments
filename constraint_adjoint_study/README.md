# Constraint sets and the adjoint of bidiagonalization: a study

**Question posed (2026-07-28).** The original bidiagonalization constraints
(paper eqs. (1)-(6), section 2.4) have dimension d_O larger than the number of
unknowns d_U = dim(L, R, B, res, c). The appendix-B.2 "unstable primal
constraints" are equivalent yet have exactly d_U equations, so their adjoint
system should determine the multipliers uniquely -- but it appears unsolvable.
What is going on conceptually? How does the stable adjoint system of the paper
relate to the adjoint of (a) the original constraints and (b) the reduced
constraints? Is there a constraint set that, when differentiated, *directly*
yields the stable adjoint system, making the augmented-Hessenberg detour
unnecessary?

## Files

| file | contents |
|---|---|
| `GUIDE_gauge_freedom.md` | Self-contained primer on gauge freedom assuming only linear algebra: underdetermined systems, the adjoint method (multiplier = weights = sensitivities), the transpose inversion, invariance lemma, four worked toy examples (incl. float32 cancellation and a 2D reprojection demo), why gauge choice governs numerics. Study-specific mapping isolated in the final dictionary section. Read this first if "gauge" is unfamiliar. Numbers verified by `guide_examples.py`. |
| `guide_examples.py` | verifies every number in the guide (gauge family, invariant gradient, float32 cancellation demos, the 2D reprojection experiment) and the solutions to all 12 exercises. |
| `NOTES_01_counting_and_gauge.md` | Dimension counts, the K^2 redundancy, multiplier gauge freedom, gradient invariance. Verified by `exp01`. |
| `NOTES_02_reduced_adjoint_is_solvable.md` | The adjoint of the reduced constraints IS uniquely solvable; explicit backward recurrence; diagnosis of why it looked unsolvable. Verified by `exp02`. |
| `NOTES_03_stable_system_is_gauge_fixing.md` | The stable adjoint system == adjoint of the ORIGINAL constraints + K^2 gauge conditions; square linear-algebra characterization without any Hessenberg reference. Verified by `exp03`, `exp04`. |
| `NOTES_04_search_for_direct_constraints.md` | Can any constraint set directly produce the stable system? Impossibility argument for same-variable square sets + projector-form experiments. Verified by `exp06`, `exp07`. |
| `FINDINGS.md` | Final synthesis, with all numerical evidence collected. |
| `common.py` | Shared constraint builders, forward solver wrapper, flatten utilities. |
| `exp01_counts_ranks_gauge.py` | ranks of constraint Jacobians, null-space dimension, gradient invariance. |
| `exp02_reduced_adjoint_recurrence.py` | dense solve + explicit recurrence for the reduced adjoint; match vs autodiff. |
| `exp03_stable_multipliers.py` | extract stable multipliers, check they satisfy original-constraint stationarity + selection identities; compare gauges. |
| `exp04_gauge_fixed_dense_solve.py` | build the square [stationarity + selection] system from the ORIGINAL constraints only; show it reproduces the stable adjoint exactly. |
| `exp05_float32_recurrence_stability.py` | float32 stability: unstable-gauge recurrence vs stable recurrence (with/without reprojection) vs autodiff. |
| `exp06_projector_constraints.py` | square Gram-Schmidt/projector constraint sets: do their adjoints coincide with the stable one? are they stable? |
| `exp07_impossibility_span.py` | numerical rank argument: no same-variable square constraint set can reproduce the stable multipliers. |
| `exp08_finalists_and_robustness.py` | exact res-block obstruction; V3 vs gauge-fixed vs paper recurrence across seeds/ensembles. |
| `exp09_partial_depth.py` | same at partial depth K = M/2 (res nonzero). |
| `exp10_error_growth_rate.py` | reduced gauge: 0.59 digits lost/iteration, precision-independent. |
| `exp11_v3_recurrence.py` | O(K) backward recurrence for the V3 adjoint; float32 parity with the paper's method. |
| `exp12_carry_correspondence.py` | band decomposition of carry differences across gauges; per-column multiplier norms. |
| `exp13_reduced_multiplier_is_intrinsically_huge.py` | 60-digit rerun of the reduced recurrence: multiplier growth is intrinsic, recurrence is forward-stable, gradient dies by cancellation. |
| `exp14_shared_error_structure.py` | float32 error decomposition: stable and V3 share the (dominant) processed primal error, private rounding independent. |
| `exp15_forward_backward_rate_match.py` | the multiplier growth rate is the backward loop gain \|A\|^2/(alpha beta); tight for Hilbert (5 digits/step, \|mu\| up to 2.6e70), not the forward drift rate. |
| `exp16_parity_proof_sparsity_theorem.py` | dense augmented Hessenberg adjoint: parity equivariance, exact checkerboard zeros, pruning onto the stable gauge (all to 1e-15/exact). |
| `NOTES_07_parity_proof_of_sparsity.md` | half-page Z2-parity proof of the paper's sparsity Theorem 4.1 (replaces the induction appendix); pruning chain closed. Verified by `exp16`. |
| `NOTES_05_direct_proof.md` | direct induction proof of existence/uniqueness for the gauge-fixed square system (no Hessenberg). |
| `NOTES_06_carry_correspondence_and_intrinsic_size.md` | carry-correspondence theorem (stable == V3 outside the pinned band, with proof); the intrinsic-size law of adjoint stability. Verified by `exp12`, `exp13`. |
| `stable_recurrence.py` | transparent reimplementation of the paper's stable backward recurrence, records multipliers. |

## Session log

- 21:26 start; folder created; writing counting notes + common utilities.
- 21:31 exp01 PASS: d_O - d_U = K^2, rank(J_full) = d_U, null dim = K^2,
  gradient gauge-invariant to 1e-15. Reduced-gauge multiplier norm >> lstsq
  norm (212 vs 31 at N7 M5 K4) -- first hint of its numerical badness.
- 21:33 exp02 PASS: reduced adjoint solved by explicit backward recurrence,
  matches dense solve and autodiff to 1e-14. Diagnosis of "cannot solve":
  B.3 with TRIANGULAR Sigma/Omega has null dim exactly K^2 - K (verified);
  with diagonal Sigma/Omega it is square nonsingular.
- 21:40 exp03 PASS: stable recurrence multipliers satisfy the ORIGINAL
  constraints' stationarity system (1e-15) + K^2+2K-1 selection identities
  (5e-16). Reduced gauge: same stationarity, selection violated by O(100),
  same gradient. Difference is a null vector.
- 21:42 exp04 PASS: square system [stationarity minus alpha/beta rows +
  selection rows], built from ORIGINAL constraints only, is nonsingular and
  reproduces the stable multipliers to 1.4e-15. No Hessenberg needed.
- 21:58 exp05: float32 recurrence stability. autodiff32 NaN everywhere;
  reduced gauge diverges exponentially EVEN IN FLOAT64; stable+reproj flat.
- 22:05 exp06: projector (Gram-Schmidt) square sets. Multipliers != stable
  (predicted); V3 (project the residuals) matches the gauge-fixed system's
  float32 accuracy and conditioning almost exactly; con_red cond explodes.
- 22:08 exp07: impossibility rank test. leftover map rank = K^2 + 2K >> 2K
  budget => no square set (plain recurrence rows + any 2K x-only rows) can
  reproduce the exact stable multipliers.
- 22:12 exp08: res-block obstruction exact ((-rho_K) - xi_K = -R eta to
  1e-15); finalists robust across seeds and a Hilbert-type ensemble.
- 22:15 exp09: partial depth K = M/2 (res != 0): same picture; reduced gauge
  NaNs on Hilbert at K = 8 already; stable trio at primal-limited floor.
- 22:20 NOTES_05: direct induction proof (existence/uniqueness of the
  gauge-fixed square system) -- removes the last conceptual need for the
  Hessenberg detour.
- 22:25 exp10: reduced-gauge error growth rate = 0.59 decimal digits lost
  per iteration (float64 and float32 parallel); 1/|beta| proxy negligible.
- 22:30 FINDINGS.md synthesis written.
- 22:40 exp11: derived and implemented the O(K) backward recurrence for the
  V3 (Gram-Schmidt) adjoint. float64: matches autodiff (3e-16) and the dense
  V3 multiplier (2e-15). float32: matches the paper's stable recurrence to
  2-3 digits at every configuration up to n = 128, K = 64. => the conjecture
  is TRUE in its achievable form: a square constraint set exists whose
  plain Lagrangian adjoint is a stable O(K) recurrence; it is provably NOT
  the paper's exact system (exp07/08), but is its numerical equal.
- 22:08 exp12: band decomposition of carry differences. RESULT:
  stable-vs-V3 agree to 5e-16 in the future band, the residual direction
  and out-of-span; differ O(1) ONLY in the past band. stable-vs-reduced
  differ O(1)-O(10) in past AND future bands. Per-column norms:
  |lam_red_n| reaches 7e10 at (32,16,16) against |grad| = 4.4e2.
- 22:12 proof found for the carry correspondence (sandwich the shared
  gradient between frame vectors; selection identities on one side,
  projectors on the other). Written up in NOTES_06 sec. 1.
- 22:14 exp13 (mpmath, 60 digits): the reduced multiplier's size is REAL
  (0.68 digits/backward step in exact arithmetic), and the float64
  recurrence is columnwise forward-stable (2e-16 relative!). exp10
  reinterpreted: gradient error = cancellation floor eps*|mu|/|grad|, not
  accumulated roundoff. Stability of a formulation = smallness of its
  multiplier. NOTES_06 sec. 2 (22:19); corrections propagated to NOTES_02
  sec. 3, exp10 docstring, FINDINGS secs. 4-7.
- 22:24 exp14: float32 error decomposition, native vs cast-down primal.
  Native primal: mutual disagreement of stable and V3 is 3x-1000x below
  their common error (0.001 on the hardest Hilbert cases -- same wrong
  answer); cast-down primal: mutual/err ~ 1 (independent, equal-sized
  private rounding). Confirms: error budget = shared processed primal
  error (identical propagation, per the theorem) + subdominant independent
  noise. Both methods are primal-limited. NOTES_06 sec. 1.1.
- 22:27 full reproducibility re-run: all 14 experiments exit OK; exp01 and
  exp04 key numbers spot-checked against the notes verbatim.
- 22:31 exp15: what sets the reduced-gauge growth rate. The naive
  "transpose = forward drift rate" hypothesis is REFUTED (normal: within
  factor 2; Hilbert: off by 14x -- adjoint 5.0 digits/step vs forward
  0.35). The correct law is the backward loop gain
  mean log10(|A|^2/(alpha beta)): 4.8-4.9 predicted vs 4.85-5.10 measured
  for Hilbert (|mu| reaches 2.6e70, mpmath-confirmed real and still
  columnwise float64-accurate), upper bound with 0.3 slack for normal.
  exp10's "divisions negligible" is normal-ensemble-specific. NOTES_06
  sec. 2, FINDINGS sec. 7 thread resolved.
- 22:44 exp16 + NOTES_07: half-page Z2-PARITY PROOF of the paper's
  sparsity Theorem 4.1. S = diag(I, -I) anticommutes with the augmented
  matrix; the Hessenberg adjoint system is equivariant under
  (Lambda, Gamma, lambda_0, gamma) -> (-S Lambda D, D Gamma D,
  -S lambda_0, -D gamma); uniqueness => fixed point => checkerboard Gamma,
  alternating gamma/Lambda, M-block lambda_0. Verified: claimed zeros
  EXACTLY 0.0 in the dense solve, equivariance residual 1e-15. Bonus: the
  pruned augmented multiplier equals the stable gauge (Phi, Xi, kappa,
  eta) to 3e-15 -- the pruning chain of NOTES_03 sec. 5 is now verified at
  every link, closing the last open thread about the detour.
- (2026-07-29, 15:20) GUIDE_gauge_freedom.md + guide_examples.py: a
  self-contained primer on gauge freedom for readers with only linear
  algebra (underdetermined systems, adjoint method, transpose inversion,
  invariance lemma with proof, three worked toy examples incl. float32
  cancellation demos, gauge-fixing criteria, dictionary into the study).
  All numbers in the guide verified by the script.
- (2026-08-09, 18:50) GUIDE_gauge_freedom.md revised for pen-and-paper
  use: all math converted from code blocks to $$LaTeX$$, explicit J_x^T
  matrices and row-by-row equation derivations added to examples A and B,
  8 tiny exercises embedded in sections 1-8 with full worked solutions in
  new section 14. guide_examples.py gained an exercises() function that
  verifies every solution (dimension counts match exp01: d_U=61, d_O=77,
  gauge dim 16; min-norm t=-1/6; float32 loss threshold |c| ~ 3/eps32).
- (2026-08-09, 20:40) GUIDE_gauge_freedom.md second revision: (i) the
  word "multiplier" now introduced explicitly -- one weight per
  constraint row, gradient = weighted sum of theta-gradients of the
  rows, mu_i = -(sensitivity of loss to violating constraint i), with a
  new exercise verifying the sensitivity reading; (ii) all study-specific
  references removed from the main flow (sections 1-13 are now fully
  generic) and collected in the expanded dictionary section 14; (iii) new
  worked example D: reprojection in 2D -- unstable iteration w -> Gw
  (eigenvalues 3, 1/3) with pinned identity l.w = 0 known in advance;
  naive float32 error grows ~eps*9^n (dead by step 8-10, rel err 4.8 at
  n=10), re-enforcing the pin each step (w -= (l.w/8) u, exact surgery on
  the unstable mode) keeps error at the precision floor (2e-6 at n=60).
  Three new exercises (eigen-structure, precision-only-postpones budget,
  exactness of the projection); 12 exercises total, all verified by
  guide_examples.py example_D() + exercises().
