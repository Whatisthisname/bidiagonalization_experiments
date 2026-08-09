#!/usr/bin/env python
"""exp07: impossibility of reproducing the EXACT stable multipliers from any
square constraint set of the form

      c~ = ( recR, recL, init, g(x) )      with g: 2K rows, x-only
           [same recurrence/init rows as con_full, any 2K extra rows]

Argument (NOTES_04): stationarity of such a set reads

      grad_x rho + J_reci^T (Xi~, Phi~, kappa~) + G^T s = 0,   G = dg/dx.

If (Xi~, Phi~, kappa~) is required to equal the stable (Xi, Phi, kappa) for
EVERY loss cotangent, then

      G^T s(grad) = leftover(grad) := -grad_x rho - J_reci^T (Xi, Phi, kappa)
                  = J_orth^T (Sigma, Omega, eta)(grad)

for every cotangent 'grad'.  The left side ranges in the FIXED subspace
range(G^T), of dimension <= 2K.  So if the span of leftover(grad) over all
cotangents has dimension > 2K, no such constraint set exists.

This experiment computes rank{ leftover(grad) : grad in basis } numerically.
Measured: exactly K^2 + 2K, with a clean spectral gap; all that matters for
the argument is that it exceeds 2K.

Same argument with g allowed to depend on th: the gradient formula acquires
-(dg/dth)^T s, which must vanish for all cotangents for the gradient to stay
correct (the recurrence/init rows already deliver the full true gradient);
since s(grad) generically spans R^{2K}, dg/dth must be 0 in the relevant
directions -- reduces to the x-only case.
"""

import numpy as np
import jax

from common import con_full, dims, flat, flatteners, forward, jacobians, random_problem
from exp03_stable_multipliers import flatten_mu_full
from exp04_gauge_fixed_dense_solve import coordinate_masks
from stable_recurrence import stable_adjoint


def run(N, M, K, seed=0):
    print(f"\n=== (N, M, K) = ({N}, {M}, {K}) ===")
    d_U, d_O = dims(N, M, K)
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)
    unflat_x, _ = flatteners(N, M, K)

    Jx, _ = jacobians(con_full, x, th, N, M, K)
    # rows of con_full: recR (MK), recL (NK), init (M) -> "reci";
    # then orthL, orthR, orthres -> "orth"
    n_reci = M * K + N * K + M
    J_reci = Jx[:n_reci, :]
    J_orth = Jx[n_reci:, :]

    lefts = []
    for j in range(d_U):
        gx_flat = np.zeros(d_U)
        gx_flat[j] = 1.0
        gx = unflat_x(gx_flat)
        m = stable_adjoint(x, th, gx, reproject=True)
        mu = flatten_mu_full(m, N, M, K)
        mu_reci, mu_orth = mu[:n_reci], mu[n_reci:]
        # leftover via the orthogonality rows (equivalent to the definition):
        lefts.append(J_orth.T @ mu_orth)
        # consistency: stationarity means J_reci^T mu_reci + J_orth^T mu_orth
        #              + gx = 0
        assert np.linalg.norm(J_reci.T @ mu_reci + J_orth.T @ mu_orth + gx_flat) < 1e-10
    Lmat = np.array(lefts).T  # d_U x d_U
    s = np.linalg.svd(Lmat, compute_uv=False)
    tol = s[0] * max(Lmat.shape) * np.finfo(np.float64).eps
    rank = int((s > tol).sum())
    print(f"rank of the leftover map = {rank}" f"  (budget of ANY square completion: 2K = {2 * K})")
    print(
        f"singular values around the threshold: "
        f"s[2K-1] = {s[2*K-1]:.2e}, s[2K] = {s[2*K]:.2e}, "
        f"s[rank-1] = {s[rank-1]:.2e}, s[rank] = {(s[rank] if rank < len(s) else float('nan')):.2e}"
    )
    verdict = "IMPOSSIBLE" if rank > 2 * K else "possible"
    print(f"=> reproducing the exact stable multipliers with a square set of " f"this class: {verdict}")


if __name__ == "__main__":
    run(7, 5, 4)
    run(6, 6, 3)
    run(5, 8, 4)
