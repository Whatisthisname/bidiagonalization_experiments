# Notes 02: The adjoint of the reduced constraints IS uniquely solvable

Numerical verification: `exp02_reduced_adjoint_recurrence.py`.

## 1. The system

Differentiate the reduced constraints (NOTES_01 sec. 3) in the standard
Lagrangian way, `Lag = rho + <mu, con_red>` with multiplier blocks

    kappa   (M,)   for  init:   r_1 - c v
    lam_n   (N,)   for  recl_n: alpha_n l_n - A r_n + beta_{n-1} l_{n-1}
    rho_n   (M,)   for  recr_n: beta_n r_{n+1} - A^T l_n + alpha_n r_n  (n<K)
                        and     res - A^T l_K + alpha_K r_K             (n=K)
    sig_n   ( )    for  nrmL_n: l_n^T l_n - 1
    omg_n   ( )    for  nrmR_n: r_n^T r_n - 1

Stationarity (0 = d Lag / d x), writing dL_n := grad_{l_n} rho etc.:

    (S-l_n)   0 = dL_n + alpha_n lam_n + beta_n lam_{n+1} - A rho_n + 2 sig_n l_n
              (lam_{K+1} := 0)
    (S-r_n)   0 = dR_n - A^T lam_n + beta_{n-1} rho_{n-1} + alpha_n rho_n
                  + 2 omg_n r_n + [n = 1] kappa       (beta_0 rho_0 := 0)
    (S-res)   0 = dres + rho_K
    (S-al_n)  0 = da_n + l_n^T lam_n + r_n^T rho_n
    (S-be_n)  0 = db_n + l_n^T lam_{n+1} + r_{n+1}^T rho_n     (n <= K-1)
    (S-c)     0 = dc - v^T kappa

and the gradient reads

    grad_A = - sum_n (lam_n r_n^T + l_n rho_n^T),      grad_v = -c kappa.

This is exactly the paper's appendix B.1 system (up to the paper writing
`l_n sigma_n` for our `2 sig_n l_n`; absorb the 2). Unknown count
NK + MK + M + 2K = d_U; equation count the same; exp01 already showed the
matrix is invertible. So the system is uniquely solvable. The only question
is HOW, i.e. in which order the equations decouple.

## 2. The explicit backward recurrence

The key device: **the scalar equations (S-al), (S-be) tell you, in advance,
the components of the unknown vectors along l_n and r_{n+1}**; use them to
eliminate the scalar multipliers before dividing.

Initialize `rho_K = -dres` (from S-res), `lam_{K+1} = 0`. For n = K, ..., 1:

1. All inner products on the right of (S-l_n) against l_n are known:
   from (S-al_n):  l_n^T lam_n = -da_n - r_n^T rho_n           ... (i)
   from (S-be_n):  l_n^T lam_{n+1} = -db_n - r_{n+1}^T rho_n   ... (ii), n<K
   Multiply (S-l_n) by l_n^T (recall l_n^T l_n = 1):
       0 = l_n^T dL_n + alpha_n (l_n^T lam_n) + beta_n (l_n^T lam_{n+1})
           - l_n^T A rho_n + 2 sig_n
   Every term except sig_n is known -> **sig_n**.
2. Then (S-l_n) itself gives
       lam_n = -( dL_n + beta_n lam_{n+1} - A rho_n + 2 sig_n l_n ) / alpha_n.
3. Same on the r side. From (S-al_n): r_n^T rho_n = -da_n - l_n^T lam_n, and
   from (S-be_{n-1}) (used at step n): r_n^T rho_{n-1} = -db_{n-1}
   - l_{n-1}^T lam_n, both known once lam_n is. Multiply (S-r_n) by r_n^T:
       0 = r_n^T dR_n - r_n^T A^T lam_n + beta_{n-1} (r_n^T rho_{n-1})
           + alpha_n (r_n^T rho_n) + 2 omg_n        -> **omg_n**
   (for n = 1 the kappa-term replaces the beta-term; use (S-c) instead:
   v^T kappa = dc with v = r_1/c, i.e. r_1^T kappa = c dc, which pins the
   r_1-component of kappa and hence omg_1.)
4. Then (S-r_n) gives rho_{n-1} (divide by beta_{n-1}), or kappa at n = 1.

Each step costs two matvecs with A -- same complexity as the paper's stable
recurrence. exp02 implements this and confirms machine-precision agreement
of all stationarity residuals and of the gradient vs autodiff.

**Conclusion: "cannot solve" is not a property of the system.** The reduced
adjoint is a perfectly solvable, uniquely determined, block-triangular
system. The two places an attempt typically derails:

(a) **Reading Sigma, Omega as (triangular) matrices.** The paper's appendix
    B.3 typesets the unstable adjoint in matrix form with terms `L Sigma`,
    `R Omega` immediately after section 4 used *triangular* Sigma, Omega. If
    Sigma, Omega are allowed to be triangular there, the system acquires
    K^2 - K spurious degrees of freedom and becomes underdetermined -- there
    is then genuinely "no way to solve it". The B.2 constraints only supply
    *diagonal* Sigma, Omega (one scalar per norm constraint). exp02 verifies:
    with diagonal Sigma/Omega the B.3 system is square and nonsingular; with
    triangular Sigma/Omega its matrix has a null space of dimension exactly
    K^2 - K.

(b) **Trying to solve (S-l_n) for lam_n before knowing sig_n** (or the
    analogous r-step). The vector equation alone is N equations for N+1
    unknowns (lam_n, sig_n); it MUST be paired with the scalar equations
    (S-al_n)/(S-be_n), which pin the l_n-component. If instead one tries to
    determine sig_n from the norm constraint "downstream", one goes in
    circles: the norm constraint's derivative information is (S-al)/(S-be)
    themselves.

## 3. Why the reduced system is nevertheless the WRONG one to solve in floats

(This section was sharpened by exp13 after exp10; the mechanism is cleaner
-- and more damning -- than "error accumulation".)

1. **The recurrence itself is fine.** In 60-digit arithmetic vs float64 on
   the same primal, every column of the float64-computed multiplier is
   RELATIVELY accurate to ~2e-16 (exp13): the backward recurrence is
   columnwise forward-stable. Nothing "goes wrong" while solving.
2. **The multiplier itself is exponentially large.** The exact solution has
   |lam_n| growing ~0.68 decimal digits per backward step (exp13, 60-digit
   arithmetic; the K-sweep fit of the resulting gradient error in exp10
   gives the same phenomenon at rate 0.59 across the ensemble): at
   (N,M,K) = (32,16,16), |lam_1| ~ 7e10 while |grad_A| ~ 4e2. Structural
   reason ("pins too little"): the system prescribes only the 2K-1 current
   inner products l_n^T lam_n, l_n^T lam_{n+1}; all other in-span components
   are whatever the coupled two-term recurrences produce, and those transfer
   operators are the (exponentially growing) inverses of the bidiagonal
   recursion -- the same growth un-reorthogonalized GKB shows forward.
   Sensitivity meaning: mu_row = dLoss/d(violation of row); the loss is
   exponentially sensitive to violations of the EARLY recurrence rows in
   this representation.
3. **So the gradient dies by cancellation, not by solver error.** Assembly
   grad_A = -(lam R^T + L rho^T) produces an O(|grad|) result from
   O(|mu|) terms: the achievable relative gradient error in precision eps is

       (eps_arith + eps_primal) * |mu|_inf / |grad|,

   verified in exp13 (predicted 3.5e-8 arithmetic / 1.6e-7 including the
   1e-15 primal orthogonality drift; observed 2.8e-7 in float64 -- and even
   the EXACT 60-digit multiplier's gradient is off by 2.3e-7, because the
   primal's imperfection is itself amplified by |mu|/|grad|). In float32
   the formula predicts total loss for K >~ 12, exactly as observed
   (exp05/exp08/exp09). The naive "small divisor" proxy
   sum log10(1/|beta_n|) is ~0.5 decades TOTAL at these sizes, i.e.
   divisions per se explain none of it. Contrast: the stable gauge and the
   V3 projector gauge keep |mu| = O(|grad|) (exp12), so their floor is
   ~eps. See NOTES_06 sec. 2 for the law and its consequences.

## 4. GKB orthogonality induction (the equivalence claim of B.2), for the record

Claim: reduced constraints + no-breakdown  =>  full orthogonality + R^T res=0.
Induction hypothesis H_n: {r_1..r_n} orthonormal, {l_1..l_{n-1}} orthonormal,
and l_j^T l_i = 0 for j < i <= n-1.

- Base: |r_1| = 1 from nrmR_1.
- l-step: alpha_n l_n = A r_n - beta_{n-1} l_{n-1}. For j < n:
  l_j^T A r_n = (A^T l_j)^T r_n = (beta_j r_{j+1} + alpha_j r_j)^T r_n
  which is 0 for j < n-1 and beta_{n-1} for j = n-1 (using H_n). Hence
  alpha_n l_j^T l_n = 0 - beta_{n-1} [j = n-1] + beta_{n-1} [j = n-1] = 0;
  divide by alpha_n != 0. Unit norm from nrmL_n.
- r-step: beta_n r_{n+1} = A^T l_n - alpha_n r_n. For j <= n:
  r_j^T A^T l_n = (A r_j)^T l_n = (alpha_j l_j + beta_{j-1} l_{j-1})^T l_n
  = alpha_n [j = n]; so beta_n r_j^T r_{n+1} = alpha_n [j=n] - alpha_n [j=n]
  = 0; divide by beta_n != 0. Unit norm from nrmR_{n+1}.
- res-step: res = A^T l_K - alpha_K r_K is the K-th r-step without the
  normalization; the same computation gives r_j^T res = 0 for all j <= K.

So the reduced set B.2 is genuinely equivalent to (1)-(6) on the
no-breakdown locus -- the user's premise 2 is verified. (Premise 1, the
counting, is verified in exp01/NOTES_01.)
