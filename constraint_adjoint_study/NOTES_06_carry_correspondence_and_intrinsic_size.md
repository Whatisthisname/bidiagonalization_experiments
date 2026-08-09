# Notes 06: the carry-correspondence theorem, and why stability = multiplier size

Numerical verification: `exp12_carry_correspondence.py`,
`exp13_reduced_multiplier_is_intrinsically_huge.py`.

These two results close the study. The first turns exp11's "V3 behaves like
the paper's recurrence" into an exact algebraic statement with a three-line
proof. The second replaces the qualitative "the reduced gauge is unstable"
with a quantitative law that explains every float32/float64 observation made
so far, and in doing so *corrects* the interpretation of exp10.

Sign conventions (as verified in exp02/exp03/exp11):

    stable  : grad_A =  Phi R^T + L Xi^T
    V3      : grad_A = -sum_n [ (I-P^L_{n-1}) lam_n r_n^T
                                + l_n ((I-P^R_n) rho_n)^T ]
    reduced : grad_A = -(lam R^T + L rho^T)

so carries are compared as Phi_n vs -lam_n and Xi_n vs -rho_n.

## 1. The carry-correspondence theorem (stable vs V3)

**Theorem.** At the exact primal (on-manifold), for every n:

    Phi_n + lam_n  \in  span(l_1, ..., l_n)
    Xi_n  + rho_n  \in  span(r_1, ..., r_{n+1})

i.e. the paper's stable carries and the V3 Gram-Schmidt carries agree
EXACTLY in every future-band component (l_j, j > n; r_j, j > n+1), in the
residual direction, and in the orthogonal complement of the computed bases.
They differ only inside the "past band" -- precisely the components that the
stable gauge pins to loss-supplied values (selection identities) and that
the V3 recurrence annihilates with projectors at every use.

**Proof.** Both multiplier sets are exact, so both gradient formulas equal
the same matrix G = dLoss/dA. Sandwich G between frame vectors and use
orthonormality of the computed bases.

(i) l-side future band: fix j > n and evaluate l_j^T G r_n twice.

    stable:  l_j^T G r_n = l_j^T Phi_n + (r_n^T Xi_j)
             and r_n^T Xi_j = 0 because n < j means (n,j) lies in the
             pinned region m <= j+1 of the selection identities
             (R^T Xi masked) with pinned value 0 (the only nonzero pin
             is at m = j, value da_j);
    V3:      l_j^T G r_n = -(l_j^T (I-P^L_{n-1}) lam_n) - r_n^T (I-P^R_n) rho_n
             = -l_j^T lam_n - 0,
             since (I-P^L_{n-1}) leaves l_j (j > n-1) untouched and
             (I-P^R_n) annihilates r_n.

    Hence l_j^T (Phi_n + lam_n) = 0 for all j > n.

(ii) r-side future band: fix j > n+1 and evaluate l_n^T G r_j twice.

    stable:  l_n^T G r_j = l_n^T Phi_j + r_j^T Xi_n, and l_n^T Phi_j = 0
             because n <= j-2 lies in the pinned region m <= j of the
             L^T Phi selection with pinned value 0 (only nonzero pin at
             m = j-1, value db_{j-1});
    V3:      l_n^T G r_j = -(l_n^T (I-P^L_{j-1}) lam_j) - r_j^T (I-P^R_n) rho_n
             = -0 - r_j^T rho_n.

    Hence r_j^T (Xi_n + rho_n) = 0 for all j > n+1.

(iii) residual direction: with u = res/|res| (u orthogonal to all r_m),
    l_n^T G u = Xi_n^T u (stable) = -u^T (I-P^R_n) rho_n = -u^T rho_n (V3).

(iv) out-of-span: (I-P^L_K) G r_n = (I-P^L_K) Phi_n (stable)
    = -(I-P^L_K) lam_n (V3), and symmetrically for the r-side with
    (I - P^{[R res]}).                                              QED

Verified in exp12 at (12,8,4), (9,6,3), (16,12,6): future band, res band and
out-of-span components of the differences all ~1e-15 while the past-band
differences are O(1) (they must be: the stable past band holds db/da pins
and zeros; the V3 past band holds whatever stationarity produced, and the
projectors make those components irrelevant).

**Reading.** The two algorithms propagate THE SAME out-of-past-band carry;
they differ only in the bookkeeping of a band that neither actually uses as
dynamical information: stable overwrites it with prescribed values
(reprojection), V3 filters it out (projectors). This is the precise sense in
which the V3 adjoint recurrence and the paper's stable recurrence are one
algorithm in two gauges -- and it explains, exactly, the float32 parity
observed in exp11 (the dynamically active components satisfy identical
recurrences; the noise entering through the inactive band is suppressed in
both, once by overwriting, once by projection).

By contrast, stable-vs-reduced differs O(1)-O(10) in the FUTURE band too
(exp12): the reduced carry is a genuinely different dynamical trajectory,
not a re-bookkeeping. (Its out-of-span part still agrees -- that much is
forced for ANY exact multiplier by argument (iv), which only uses the
gradient formulas.)

### 1.1 Floating-point coda (exp14)

Decompose each method's float32 gradient error into a common and an
individual part by comparing the two methods against the float64 truth AND
against each other, under two primals:

- **native float32 primal** (realistic use): mutual disagreement is 3x-1000x
  SMALLER than either method's error (ratio 0.00-0.67, median ~0.15; on the
  hardest Hilbert instances, where both err by 1e-2, the ratio drops to
  0.001 -- they return nearly the SAME wrong gradient). The dominant error
  is common-mode: the primal's imperfection, amplified through the
  propagation that the theorem says is identical in both methods.
- **float64 primal cast to float32** (primal error suppressed to eps):
  mutual/err ~ 0.5-1.5. The methods' own local rounding is independent and
  of equal size; neither is intrinsically more accurate.

So the exp11 observation "V3 matches the paper's recurrence to 2-3 digits
in float32" is explained, and sharpened: the two methods do NOT commit the
same local rounding errors -- they share the (dominant) processed primal
error because their propagated carries are identical, and their private
arithmetic noise is equal-sized and subdominant. Practically: the two are
interchangeable, and BOTH are primal-limited; improving the adjoint further
requires improving the forward pass, not the backward one.

## 2. The intrinsic-size law (why the reduced gauge cannot work numerically)

exp13 reruns the reduced backward recurrence in 60-digit arithmetic
(mpmath) on the same float64 primal, at (N,M,K) = (32,16,16):

| n | true |lam_n| (60-digit) | float64 rel. err |
|---|---|---|
| 1 | 2.4e10 | 2.5e-15 |
| 4 | 6.3e10 | 2.9e-16 |
| 8 | 7.8e08 | 2.1e-16 |
| 12 | 6.1e05 | 1.7e-16 |
| 16 | 4.5e00 | 2.0e-16 |

Two findings, one of which corrects an earlier reading:

1. **The true reduced multiplier is exponentially large**: 0.68 decimal
   digits of growth per backward step (fit over n = 16..1), while the
   gradient it encodes stays |grad_A| = 4.4e2. This is a property of the
   exact linear algebra, not of floating point.
2. **The reduced recurrence is columnwise forward-stable**: every column is
   computed to ~2e-16 RELATIVE accuracy in float64. There is NO error
   accumulation in the recurrence at all. (exp10's "0.59 digits per step of
   error growth" is therefore not accumulating roundoff -- it is the growth
   of the multiplier itself with K, propagated into the gradient through
   the final assembly; see the floor formula below.)

The gradient is assembled as grad_A = -(lam R^T + L rho^T), an O(|grad|)
result summed from O(|mu|) terms. The cancellation depth is

    log10( |mu|_inf / |grad| ) = log10( 7.1e10 / 4.4e2 ) = 8.2 digits,

so the best any float64 evaluation THROUGH THIS REPRESENTATION can do is

    rel. gradient error  >~  (eps_arith + eps_primal) * |mu|_inf / |grad|

with eps_primal ~ 1e-15 the orthogonality drift of the computed primal
(which enters the on-manifold identities the recurrence uses, again scaled
by the multiplier). Predicted floor 3.5e-8 (arithmetic) to 1.6e-7 (primal);
observed float64 error 2.8e-7; the 60-digit multiplier's gradient, where
cancellation is harmless, still differs from the stable gradient by 2.3e-7
-- i.e. even EXACT arithmetic cannot rescue the reduced gauge at the float64
primal: the primal's 1e-15 imperfection alone, amplified by |mu|/|grad|,
already costs 8 digits. In float32 the same law gives
1.2e-7 * 1.6e8 >> 1: total loss, exactly as observed in exp05/exp08/exp09.

**The law.** For a square (gauge-free) constraint representation, the
multiplier is unique, and

    - |mu| / |grad| is the intrinsic cancellation/conditioning factor of
      the representation;
    - the reduced (B.2) representation has |mu|/|grad| growing ~10^{0.6 K}
      (rate problem-dependent; 0.59-0.68 across our ensembles);
    - the V3 projector representation keeps |mu| = O(|grad|) (exp12 table:
      per-column norms 1.4e0..2e2 against |grad_A| = 4.4e2, matching the
      stable gauge's carries column by column outside the pinned band);
    - the paper's stable selection picks, out of the K^2-dimensional
      multiplier family of the redundant representation, a member with
      |mu| = O(|grad|).

Multiplier size has a standard sensitivity meaning: mu_row measures
dLoss/d(violation of that constraint row). The reduced representation is
one in which the loss is exponentially sensitive to violations of the early
recurrence rows -- a perturbation of row n must be carried through all
remaining steps to be felt. The projector rows of V3 (and the paper's
reprojection) make the same information available locally, so no row needs
a large multiplier. "Stability of the adjoint" is therefore not about HOW
the multiplier system is solved (the unstable-gauge recurrence solves its
system essentially perfectly); it is about WHICH representation's
multiplier one insists on computing.

**What sets the rate (exp15).** The growth rate of |lam_n| per backward
step is the loop gain of the backward two-term recursion,

    rate  <=  mean_n log10( |A|_2^2 / (alpha_n beta_n) ),

(two A-applications, one alpha- and one beta-division per full step). For
the Hilbert-type ensemble this bound is TIGHT: alpha, beta sit at the
noise floor ~3e-3, |A| ~ 2, predicted 4.8-4.9 vs measured 4.85-5.10 digits
per step -- the exact multiplier reaches 2.6e70 (!) at (32,16,16),
confirmed at 60-digit precision, with float64 still columnwise accurate to
3e-14. For the normal ensemble the bound (0.87-0.97) has ~0.3 digits/step
of slack over the measured 0.49-0.68 (operator norms overestimate the gain
along the trajectory). Two hypotheses this KILLS: (i) "the rate equals the
orthogonality-drift rate of un-reorthogonalized forward GKB" -- true within
a factor ~2 for the normal ensemble (0.63-1.10 forward), but off by 14x for
Hilbert (forward drift 0.30-0.44/step, capped by saturation/convergence;
the adjoint gain has no such cap); (ii) "beta-divisions alone explain it"
(exp10's negligible-proxy finding is normal-ensemble-specific; for Hilbert
the divisions supply 2.2 of the ~5.0, the alpha-divisions and A-couplings
the rest).

## 3. Consequences for the study's conclusions

- The right abstract summary of the paper's construction: among all
  multiplier representations of d Loss/d theta for GKB, choose one whose
  multiplier stays O(|grad|); the selection identities (equivalently the
  augmented-Hessenberg pruning, equivalently the V3 projector adjoint
  modulo the pinned band) all land on such a representative, and they agree
  with each other in every dynamically-propagated component (sec. 1).
- exp10 should be read as measuring d log10 |mu_red| / dK, not roundoff
  accumulation.
- Any future "direct constraint set" proposal can be screened cheaply:
  compute its unique multiplier at small size and check |mu|/|grad| growth
  in K. No float32 experiments needed for a first verdict.
