# Notes 05: A direct existence/uniqueness proof for the stable system (no Hessenberg)

NOTES_03 established numerically that the square system

    [ stationarity of con_full, minus the d/dalpha and d/dbeta rows ]
    [ selection rows:  (L^T Phi)_{mn} = db_{n-1}[m=n-1]      for m <= n
                       (R^T Xi)_{mn}  = da_n   [m=n]         for m <= n+1 ]

is nonsingular and reproduces the stable multipliers. Here is a direct proof
by induction -- the same back-substitution order as the paper's backward
recurrence -- valid on the no-breakdown locus. Everything is evaluated at an
exact solution (L, R orthonormal, alpha_n != 0, beta_n != 0, c != 0). This
removes the last conceptual role of the augmented-Hessenberg detour: it is
not needed for the statement, the derivation, the solvability proof, or the
algorithm; it remains a (nice) alternative derivation and the historical
discovery route.

Recall the unknowns mu = (Xi, Phi, kappa, Sigma, Omega, eta), with Sigma,
Omega lower triangular, and the stationarity rows (NOTES_03 sec. 1):

    (SL_n)  0 = dL_n + A xi_n - (Phi B^T) e_n + L (Sigma + Sigma^T) e_n
    (SR_n)  0 = dR_n + A^T phi_n - (Xi B) e_n + R (Omega + Omega^T) e_n
                 + [n=1] kappa + res eta_n
    (Sres)  0 = dres - xi_K + R eta
    (Sc)    0 = dc - v^T kappa

**Step 0 (eta and xi_K).** Unknowns eta (K), xi_K (M). Equations: (Sres)
(M rows) + the selection rows for column K of R^T Xi (K rows,
r_m^T xi_K = da_K [m=K]). Substituting xi_K = dres + R eta into the
selection rows gives (R^T R) eta = da_K e_K - R^T dres. Since R^T R = I,
eta is determined (and equals the paper's eta); then xi_K. Unique.

**Step n (descending, L side).** Known: xi_n, phi_{n+1}, rows m > n of
Sigma. Unknowns: phi_n (N) and row n of Sigma (n entries Sigma_{n,m},
m <= n). Equations: (SL_n) (N rows) and the selection rows for column n of
L^T Phi (n rows). Note (Sigma + Sigma^T) e_n = (row n of Sigma, transposed)
+ (column n of Sigma), and column n entries Sigma_{m,n} (m > n) belong to
rows already computed. Write s := Sigma row n as an n-vector (padded), and
t_n := the known part of L (Sigma + Sigma^T) e_n from future rows. Then

    (SL_n):   alpha_n phi_n = dL_n + A xi_n + L_{1:n} s + t_n - beta_n phi_{n+1}

(with the convention Sigma_{nn} counted once here and once in the transpose;
bookkeeping as in the recurrence). Apply L_{1:n}^T and use L^T L = I:

    alpha_n (L_{1:n}^T phi_n) = L_{1:n}^T (known) + s.

The left side is prescribed by the selection rows (targets:
db_{n-1} e_{n-1}, zeros elsewhere). Hence s is determined explicitly, then
phi_n = (...) / alpha_n. The (N+n) x (N+n) block is invertible because its
Schur complement w.r.t. phi_n is the leading n x n Gram matrix of L (= I);
the only divisions are by alpha_n != 0. Unique.

**Step n (R side).** Same argument transposed: unknowns xi_{n-1} (M) and
row n of Omega (n entries); equations (SR_n) (M rows) + selection rows for
column n-1 of R^T Xi (n rows); divisions by beta_{n-1} != 0; Gram matrix
R^T R = I. At n = 1 the unknowns are kappa (M) and Omega_{11}; the equations
are (SR_1) (M rows) + (Sc) (1 row); the elimination needs
v^T r_1 = |r_1|^2 / c = 1/c != 0. Unique.

**Conclusion.** The square system has exactly one solution, constructed by
the recurrence above -- which is precisely the paper's stable backward
recurrence (the s-elimination is the Sigma-row formula; enforcing the
selection rows on the carries is the reprojection). Moreover the dropped
d/dalpha_n, d/dbeta_n rows hold automatically: the selection rows give
(L^T Phi)_{nn} = 0, (R^T Xi)_{nn} = da_n, (L^T Phi)_{n,n+1} = db_n,
(R^T Xi)_{n+1,n} = 0, and the alpha/beta stationarity rows are exactly the
sums of these pairs. Hence the solution satisfies the FULL stationarity
system of the original constraints and therefore delivers the exact gradient
(NOTES_01 sec. 4). QED.

Two remarks.

1. The proof uses orthonormality of the computed bases (Gram = I) -- i.e.
   the redundant constraints themselves -- to invert the per-step blocks.
   This is the precise sense in which "the K^2 redundancy buys stability":
   the selection rows are only consistent because the constraint set is
   redundant (gauge exists), and only useful because the Gram matrices are
   identity (perfectly conditioned) on-manifold.
2. In floating point the Gram matrices are I + O(eps_fwd); the per-step
   blocks remain well-conditioned, which is why enforcing the selection
   mid-flight (reprojection) is cheap and harmless. In the reduced gauge
   there are no selection rows to enforce; the analogous components of the
   multiplier come out of the recurrence itself and are intrinsically
   exponentially large (exp13: |mu| ~ 10^{0.6 K} while the gradient stays
   O(1)), which caps any finite-precision gradient at eps |mu|/|grad| --
   the exponential-in-K loss measured in exp05/exp10. See NOTES_06 sec. 2.
