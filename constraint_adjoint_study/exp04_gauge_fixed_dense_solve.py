#!/usr/bin/env python
"""exp04: the stable adjoint can be obtained DIRECTLY from the original
constraints -- no Hessenberg augmentation anywhere.

Construction (all objects built from con_full only):
  unknowns   mu in R^{d_O}  (Xi, Phi, kappa, Sigma-tril, Omega-tril, eta)
  equations  (a) stationarity rows J_full^T mu = -grad_x rho for every x
                 coordinate EXCEPT the alpha and beta coordinates
                 (d_U - (2K-1) rows),
             (b) selection rows
                 l_m^T phi_n = db_{n-1}[m=n-1]   for m <= n
                 r_m^T xi_n  = da_n   [m=n]      for m <= n+1
                 (K^2 + 2K - 1 rows).
  total rows (d_U - 2K + 1) + (K^2 + 2K - 1) = d_U + K^2 = d_O  -> square.

Claims tested (see NOTES_03):
  C1  The square matrix is nonsingular (rank d_O); report its conditioning.
  C2  Its unique solution equals the stable recurrence's multiplier vector.
  C3  The dropped alpha/beta stationarity rows are automatically satisfied
      (they are implied by the selection rows).
  C4  Gradient from this solve matches autodiff.
"""

import numpy as np
import jax

from common import (
    autodiff_reference_grad,
    con_full,
    dims,
    flat,
    flatteners,
    forward,
    jacobians,
    make_linear_loss,
    random_problem,
    rel_err,
)
from exp03_stable_multipliers import flatten_mu_full
from stable_recurrence import stable_adjoint


def coordinate_masks(N, M, K):
    """Boolean masks over the flat x vector for each variable block."""
    unflat_x, _ = flatteners(N, M, K)
    masks = {}
    for key in ["L", "R", "alphas", "betas", "res", "c"]:
        probe = dict(
            L=np.zeros((N, K)),
            R=np.zeros((M, K)),
            alphas=np.zeros(K),
            betas=np.zeros(K - 1),
            res=np.zeros(M),
            c=np.zeros(()),
        )
        probe[key] = probe[key] + 1.0
        masks[key] = np.asarray(flat(probe)) > 0.5
    return masks


def selection_matrix(x, gx, N, M, K):
    """Rows act on mu (flattened as in exp03). Returns (S, targets)."""
    L, R = np.asarray(x["L"]), np.asarray(x["R"])
    da, db = np.asarray(gx["alphas"]), np.asarray(gx["betas"])
    d_U, d_O = dims(N, M, K)
    # offsets inside mu: [Xi (MK) | Phi (NK) | kappa (M) | Sig tril | Omg tril | eta (K)]
    off_Xi, off_Phi = 0, M * K
    rows, tgts = [], []
    for n in range(K):
        for mm in range(n + 1):  # L side, m <= n
            row = np.zeros(d_O)
            # l_mm^T phi_n : phi_n occupies Phi[:, n] -> flat indices
            # Phi raveled C-order (N, K): entry (i, n) at off_Phi + i*K + n
            row[off_Phi + np.arange(N) * K + n] = L[:, mm]
            rows.append(row)
            tgts.append(db[n - 1] if (n >= 1 and mm == n - 1) else 0.0)
        for mm in range(min(n + 1, K - 1) + 1):  # R side, m <= n+1
            row = np.zeros(d_O)
            row[off_Xi + np.arange(M) * K + n] = R[:, mm]
            rows.append(row)
            tgts.append(da[n] if mm == n else 0.0)
    return np.array(rows), np.array(tgts)


def run(N, M, K, seed=0):
    print(f"\n=== (N, M, K) = ({N}, {M}, {K}) ===")
    d_U, d_O = dims(N, M, K)
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)
    loss, _ = make_linear_loss(N, M, K)
    unflat_x, _ = flatteners(N, M, K)
    gx_flat = np.asarray(jax.grad(lambda xf: loss(unflat_x(xf)))(flat(x)))
    gx = unflat_x(gx_flat)

    Jx, Jth = jacobians(con_full, x, th, N, M, K)
    masks = coordinate_masks(N, M, K)
    keep = ~(masks["alphas"] | masks["betas"])  # drop alpha/beta rows
    Astat = Jx.T[keep, :]
    bstat = -gx_flat[keep]
    Ssel, tsel = selection_matrix(x, gx, N, M, K)
    Big = np.vstack([Astat, Ssel])
    rhs = np.concatenate([bstat, tsel])
    print(f"C1  system shape {Big.shape} (d_O = {d_O})")
    s = np.linalg.svd(Big, compute_uv=False)
    print(f"C1  nonsingular: smallest sv {s[-1]:.2e}, cond {s[0]/s[-1]:.2e}")

    mu = np.linalg.solve(Big, rhs)

    m_stable = stable_adjoint(x, th, gx, reproject=True)
    mu_stable = flatten_mu_full(m_stable, N, M, K)
    print(f"C2  |mu_square_solve - mu_stable_recurrence| / |mu_stable| = " f"{rel_err(mu, mu_stable):.2e}")

    dropped = Jx.T[~keep, :] @ mu + gx_flat[~keep]
    print(f"C3  dropped alpha/beta stationarity residual: {np.abs(dropped).max():.2e}")

    gref = autodiff_reference_grad(A, v, K, loss)
    print(f"C4  gradient vs autodiff: {rel_err(Jth.T @ mu, np.asarray(flat(gref))):.2e}")


if __name__ == "__main__":
    run(7, 5, 4)
    run(6, 6, 3)
    run(5, 8, 4)
