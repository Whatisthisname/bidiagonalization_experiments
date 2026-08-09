# Notes 04: Is there a constraint set whose adjoint IS the stable system?

Numerical verification: `exp06_projector_constraints.py`,
`exp07_impossibility_span.py`, `exp08_finalists_and_robustness.py`,
`exp09_partial_depth.py`.

The question splits in two, and the two halves have opposite answers.

## A. Reproducing the EXACT stable system: impossible for square sets (in the original variables)

Setting: a square constraint set c~(x, th) = 0 in the original variables
whose stationarity system is solved by exactly the stable multipliers
(Phi, Xi, kappa) for every loss.

**A1. The res-block obstruction (exact, verified).** Suppose res enters c~
only through the last recurrence row `res - A^T l_K + alpha_K r_K` (true for
the reduced set and for every Gram-Schmidt/projector variant below -- there
is nowhere else for res to appear if the rows are recurrence-shaped). The
res-coordinate stationarity rows then read `dres + rho~_K = 0`, forcing
rho~_K = -dres. The stable gauge instead has xi_K = dres + R eta with
eta = da_K e_K - R^T dres != 0 generically. exp08(a) verifies
`(-rho~_K) - xi_K = -R eta` to 1e-15 for all four square sets tested, with
|R eta| = O(1). So none of them can reproduce the stable multipliers; the
gap is exactly the R^T res = 0 multiplier term, which a square set does not
possess.

**A2. The span argument (rank-verified).** More generally, keep the plain
recurrence + init rows (they are forced if the gradient formula is to read
grad_A = Phi R^T + L Xi^T with the stable Phi, Xi) and allow ANY 2K
additional x-only rows g(x) = 0 to complete the square count. Stationarity:

    grad_x rho + J_reci^T (Xi, Phi, kappa) + G^T s = 0,  G = dg/dx fixed.

If (Xi, Phi, kappa) are the stable ones, then G^T s(grad) must equal

    leftover(grad) := J_orth^T (Sigma, Omega, eta)(grad)

for every cotangent grad, i.e. the range of the leftover map must fit inside
range(G^T), whose dimension is at most 2K. exp07 computes the rank of the
leftover map over a full cotangent basis: it is K^2 + 2K (clean spectral
gap: sigma ~ 1.0 then 7e-16), strictly larger than 2K for every K >= 1.
**Verdict: impossible.** (If g is allowed to depend on th, its gradient
contribution -(dg/dth)^T s must vanish identically for the gradient to
remain correct, since the reci rows already produce the full gradient;
s(grad) generically spans R^{2K}, forcing dg/dth = 0 in all relevant
directions -- reducing to the x-only case.)

Interpretation: the stable multiplier genuinely uses K^2 + 2K "orthogonality
forces" (Sigma, Omega, eta); a square set only has 2K norm-type rows to
generate them. The deficit K^2 is precisely the gauge dimension. To get the
exact stable system from stationarity alone one must either
  (i) keep the redundant constraint set and ADD the K^2 selection rows
      (exp04 -- the honest "direct route", no Hessenberg needed), or
 (ii) lift to a larger variable space where the orthogonality constraints
      stop being redundant -- which is exactly what the augmented Hessenberg
      construction is. In the lifted space the K^2 previously-redundant rows
      pin the structural-zero blocks of the lifted basis, the system becomes
      square, the multiplier becomes unique, and pruning it back down lands
      exactly on the stable selection (NOTES_03 sec. 5).

## B. Reproducing the stable BEHAVIOR: possible, and easy

The impossibility concerns the exact multiplier values. Numerical stability
is a property of the *algorithm/system*, not of the multiplier point, and
here the answer flips.

Square Gram-Schmidt ("projector") constraint sets, all on-manifold
equivalent to bidiagonalization (residuals ~1e-15 at the computed solution):

    V1 gs-full :  alpha_n l_n = (I - L_<n L_<n^T) A r_n
                  beta_n r_{n+1} = (I - R_<=n R_<=n^T) A^T l_n
    V2 gs-half :  l-side as V1; r-side projects only past R_<n, keeps
                  -alpha_n r_n explicit
    V3 gs-resid:  projectors applied to the CLASSIC residuals:
                  alpha_n l_n = (I - L_<n L_<n^T)(A r_n - beta_{n-1} l_{n-1})
                  beta_n r_{n+1} = (I - R_<=n R_<=n^T)(A^T l_n - alpha_n r_n)

plus unit norms and the init row. Findings (exp06, exp08, exp09):

1. Their unique multipliers do NOT equal the stable ones (rel. differences
   O(0.2)-O(2); selection identities violated) -- consistent with part A.
2. Conditioning of the adjoint matrix J^T at the computed primal, n = 16/32/48
   (normal ensemble, K = n/2):
       con_red: 1.8e5 -> 5.4e12 -> 1.5e19     (exponential in K)
       V1:      2.2e3 -> 1.2e8  -> 4.3e9
       V2:      4.8e3 -> 1.7e8  -> 6.1e9
       V3:      5.4e2 -> 6.7e3  -> 1.1e5
       gauge-fixed square (exp04): 7.2e2 -> 8.2e3 -> 1.5e5
   V3 tracks the gauge-fixed stable system; the raw-vector projections
   (V1/V2) are intermediate; no projectors is catastrophic.
3. float32 dense solves at the float32 primal (errors vs float64 reference),
   across sizes 32..64, seeds 0..2, normal AND Hilbert-type ensembles, at
   full depth (exp08) and half depth K = M/2 (exp09): V3-dense, gauge-fixed
   dense, and the paper's stable recurrence agree with each other to 2-3
   digits at an error floor set by the float32 PRIMAL (1e-5..1e-3, one hard
   Hilbert instance 1e-2 where all three degrade identically), while the
   reduced recurrence/dense solve is wrong by many orders of magnitude or NaN.

Why V3 behaves like the stable gauge (mechanism, qualitative): its
constraint function differentiates the projectors. The extra Jacobian terms
are of the form L (mask o (...)) and R (mask o (...)) -- the same
"triangular-multiplier" structures L(Sigma + Sigma^T), R(Omega + Omega^T)
that the stable system introduces via the orthogonality multipliers. A
square projector set is therefore *self-gauge-fixing*: stationarity itself
manufactures the L- and R-supported correction terms, and the multiplier
components along the computed bases stop being propagated by bare divisions.
V3 (projecting the residual, norm ~ alpha_n) keeps these correction terms
small; V1/V2 (projecting A r_n, norm ~ |A|) makes them large, which shows up
directly in the conditioning.

This was later made EXACT (carry-correspondence theorem, NOTES_06 sec. 1,
verified in exp12): the V3 carries and the paper's stable carries satisfy

    Phi_n + lam_n \in span(l_1..l_n),   Xi_n + rho_n \in span(r_1..r_{n+1}),

i.e. they agree identically in every future-band, residual-direction and
out-of-span component, and differ only inside the past band that the stable
gauge pins by selection and the V3 recurrence annihilates by projectors.
The two methods are one algorithm in two gauges of the pinned band. (The
reduced gauge's carry differs O(1)-O(10) in the FUTURE band as well -- a
genuinely different trajectory.)

**B.1 The V3 adjoint as an O(K)-per-step recurrence (exp11).** Deriving the
V3 stationarity on-manifold gives a backward recurrence of exactly the
paper's cost class (two matvecs per step + O((N+M)K) projector/accumulator
work):

    rho_K = -dres;    for j = K..1:
      sig_j from the a_j pin  l_j^T lam_j = -da_j    (projectors kill all
                                                      other in-span terms)
      lam_j = [ -dL_j + A (I-P^R_j) rho_j - 2 sig_j l_j
                - b_j (I-P^L_j) lam_{j+1} - sum_{m>j} a_m (l_j^T lam_m) l_m ] / a_j
      omg_j from the b_{j-1} pin  r_j^T rho_{j-1} = -db_{j-1}
      rho_{j-1} = [ -dR_j + A^T (I-P^L_{j-1}) lam_j - a_j (I-P^R_j) rho_j
                    - 2 omg_j r_j - sum_{m>=j} b_m (r_j^T rho_m) r_{m+1} ] / b_{j-1}
      (kappa and the d/dc row replace the beta-division at j = 1)
    grad_A = - sum_n [ (I-P^L_{n-1}) lam_n r_n^T + l_n ((I-P^R_n) rho_n)^T ]

Note what happened: the PROJECTORS arise from differentiation (they are not
imposed), the reprojection TARGETS of the paper (da, db) arise as the
alpha/beta stationarity rows, and the carried multipliers enter only in
projected form, so in-span error components of the carries are annihilated
at every step -- reprojection with implicit targets. The future-coupling
sums are maintained as running inner-product tables (like the paper's
Sigma, Omega rows).

Results (exp11): float64 -- gradients match autodiff to 3e-16 and the
multiplier matches the dense solve of the full unsimplified V3 Jacobian to
2e-15 (the on-manifold simplification is exact at float64 primal quality).
float32, sizes n = 32..128 (K up to 64), normal and Hilbert-type ensembles,
full and half depth, multiple seeds: the V3 recurrence matches the paper's
stable recurrence to 2-3 significant digits at every single configuration
(both at the primal-limited floor, including the hard Hilbert instances
where both sit at 1e-2 together).

## C. Net answers to the original questions

1. "Is there a set of constraints that, when differentiated, lead to the
   exact stable adjoint system?" -- **Not by stationarity of any square set
   in the original variables** (A1/A2). The exact stable system is
   stationarity of the ORIGINAL (redundant) set plus K^2 selection rows
   (exp04), or equivalently the pruned adjoint of the lifted square
   Hessenberg set. The selection rows are not stationarity conditions of
   anything; they are a gauge choice, and they depend on the cotangent
   (da, db), which is something a constraint set cannot encode.
2. "Would such a find invalidate the paper?" -- It reframes rather than
   invalidates: (i) the paper's OWN system needs no Hessenberg detour to be
   *stated, derived, proved solvable, and solved* (NOTES_03 + NOTES_05); the
   detour remains a clean alternative proof and the discovery route.
   (ii) The V3 Gram-Schmidt set is a square constraint set whose textbook
   Lagrangian adjoint -- unique multiplier, no gauge choice, no selection
   rules -- yields an O(K) backward recurrence that matches the paper's
   method's float32 accuracy across every configuration tested (exp11), and
   whose carries agree with the paper's EXACTLY outside the pinned band
   (theorem, NOTES_06 sec. 1). That IS the "direct path" conjectured in the
   question, in its only possible form: it cannot and does not reproduce
   the paper's exact multiplier (part A), but it is the same propagation
   with different bookkeeping of the band neither method propagates. The
   paper's framing "the augmented detour is how one obtains a stable
   adjoint" should be softened accordingly; its system remains a canonical
   choice (the unique gauge pinning ALL in-span multiplier components to
   loss-supplied values, with reprojection as explicit re-enforcement).
