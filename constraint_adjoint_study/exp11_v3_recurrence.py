#!/usr/bin/env python
"""exp11: an O(K)-style backward recurrence for the V3 (Gram-Schmidt) adjoint.

Derivation (on-manifold; NOTES_04 sec. B gets a constructive upgrade):

Stationarity of the V3 Lagrangian, evaluated at the exact solution, gives
for j = K..1 (with rho_K = -dres, S-sums over already-computed columns):

  l_j:  0 = dL_j + a_j lam_j + b_j (I-P^L_j) lam_{j+1} - A (I-P^R_j) rho_j
            + 2 sig_j l_j + sum_{m>j} a_m (l_j^T lam_m) l_m
  r_j:  0 = dR_j - A^T (I-P^L_{j-1}) lam_j + b_{j-1} rho_{j-1}
            + a_j (I-P^R_j) rho_j + 2 omg_j r_j
            + sum_{m>=j} b_m (r_j^T rho_m) r_{m+1}   (b_K r_{K+1} == res)
            + [j=1] kappa
  a_j:  0 = da_j + l_j^T lam_j            <- direct pin (projector kills rho)
  b_j:  0 = db_j + r_{j+1}^T rho_j        <- direct pin
  c:    0 = dc - v^T kappa
  grad_A = - sum_n [ (I-P^L_{n-1}) lam_n r_n^T + l_n ((I-P^R_n) rho_n)^T ]
  grad_v = -c kappa

Each step: two matvecs with A plus O((N+M)K) projector/accumulator work --
the same cost class as the paper's stable recurrence.  The projectors and
the pins arise from DIFFERENTIATION of the Gram-Schmidt constraints; nothing
is imposed by hand.

Tests:
  C1  float64: gradients match autodiff/stable reference to ~1e-13.
  C2  float64: multiplier matches the dense solve of the (full, on-manifold
      unsimplified) V3 Jacobian to ~1e-12.
  C3  float32 sweep (normal + hilbert, full + partial depth): error at the
      primal-limited floor, tracking the paper's stable recurrence.
"""

import numpy as np
import jax

from common import flat, flatteners, forward, jacobians, make_linear_loss, random_problem, rel_err
from exp05_float32_recurrence_stability import gkb_forward, make_cotangent, cast
from exp06_projector_constraints import con_proj_factory
from stable_recurrence import stable_adjoint
import jax.numpy as jnp


def v3_adjoint_recurrence(x, th, gx, dtype=np.float64):
    L = np.asarray(x["L"], dtype)
    R = np.asarray(x["R"], dtype)
    al = np.asarray(x["alphas"], dtype)
    be = np.asarray(x["betas"], dtype)
    res = np.asarray(x["res"], dtype)
    c = dtype(np.asarray(x["c"]))
    A = np.asarray(th["A"], dtype)
    N, K = L.shape
    M = R.shape[0]
    dL = np.asarray(gx["L"], dtype)
    dR = np.asarray(gx["R"], dtype)
    da = np.asarray(gx["alphas"], dtype)
    db = np.asarray(gx["betas"], dtype)
    dres = np.asarray(gx["res"], dtype)
    dc = dtype(np.asarray(gx["c"]))

    Rnext = np.concatenate([R[:, 1:], res[:, None]], axis=1)
    benext = np.concatenate([be, np.ones(1, dtype)])

    lam = np.zeros((N, K), dtype)
    rho = np.zeros((M, K), dtype)
    sig = np.zeros(K, dtype)
    omg = np.zeros(K, dtype)
    U = np.zeros((K, K), dtype)  # U[:, m] = L^T lam_m
    V = np.zeros((K, K), dtype)  # V[:, m] = R^T rho_m
    gA = np.zeros((N, M), dtype)

    rho[:, K - 1] = -dres
    V[:, K - 1] = R.T @ rho[:, K - 1]

    for i in range(K - 1, -1, -1):
        l_i, r_i = L[:, i], R[:, i]
        rho_i = rho[:, i]
        # (I - P^R_{i+1}) rho_i   -- projector past r_1..r_{i+1}
        pr_rho = rho_i - R[:, : i + 1] @ (R[:, : i + 1].T @ rho_i)
        q = A @ pr_rho
        sig[i] = 0.5 * (-(l_i @ dL[:, i]) + l_i @ q + al[i] * da[i])
        # future-coupling sum for the l equation
        s = np.zeros(K, dtype)
        if i < K - 1:
            s[i + 1 :] = al[i + 1 :] * U[i, i + 1 :]
        Slam = L @ s
        vec = -dL[:, i] + q - 2.0 * sig[i] * l_i - Slam
        if i < K - 1:
            lam_next = lam[:, i + 1]
            pr_lam_next = lam_next - L[:, : i + 1] @ (L[:, : i + 1].T @ lam_next)
            vec = vec - be[i] * pr_lam_next
        lam[:, i] = vec / al[i]
        U[:, i] = L.T @ lam[:, i]

        # (I - P^L_{i}) lam_i  -- projector past l_1..l_i (j-1 = i columns)
        pr_lam = lam[:, i] - L[:, :i] @ (L[:, :i].T @ lam[:, i])
        p = A.T @ pr_lam
        # future-coupling sum for the r equation (m >= i)
        t = benext * V[i, :]
        t[:i] = 0.0
        Srho = Rnext @ t
        if i >= 1:
            omg[i] = 0.5 * (-(r_i @ dR[:, i]) + r_i @ p + be[i - 1] * db[i - 1])
            rho[:, i - 1] = (-dR[:, i] + p - al[i] * pr_rho - 2.0 * omg[i] * r_i - Srho) / be[i - 1]
            V[:, i - 1] = R.T @ rho[:, i - 1]
        else:
            omg[0] = 0.5 * (-(r_i @ dR[:, i]) + r_i @ p - c * dc)
            kappa = -dR[:, i] + p - al[i] * pr_rho - 2.0 * omg[0] * r_i - Srho
        gA -= np.outer(pr_lam, r_i) + np.outer(l_i, pr_rho)

    grad_v = -c * kappa
    return dict(lam=lam, rho=rho, sig=sig, omg=omg, kappa=kappa, grad_A=gA, grad_v=grad_v)


def validate_float64(N, M, K, seed=0):
    print(f"--- float64 validation, (N,M,K)=({N},{M},{K}) ---")
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)
    loss, _ = make_linear_loss(N, M, K)
    unflat_x, _ = flatteners(N, M, K)
    gx_flat = np.asarray(jax.grad(lambda xf: loss(unflat_x(xf)))(flat(x)))
    gx = unflat_x(jnp.asarray(gx_flat))

    m = v3_adjoint_recurrence(x, th, gx)
    ref = stable_adjoint(x, th, gx, reproject=True)
    print(
        f"C1  grad_A err {rel_err(m['grad_A'], ref['grad_A']):.2e}, "
        f"grad_v err {rel_err(m['grad_v'], ref['grad_v']):.2e}"
    )

    Jx, Jth = jacobians(con_proj_factory("V3"), x, th, N, M, K)
    mu_dense = np.linalg.solve(np.asarray(Jx).T, -gx_flat)
    o = M
    lam_d = mu_dense[o : o + N * K].reshape(N, K)
    o += N * K
    rho_d = mu_dense[o : o + M * K].reshape(M, K)
    print(
        f"C2  |lam_rec - lam_dense|/|lam_dense| = {rel_err(m['lam'], lam_d):.2e}, "
        f"|rho_rec - rho_dense|/|rho_dense| = {rel_err(m['rho'], rho_d):.2e}"
    )


def float32_sweep():
    print("\n--- float32 sweep: V3 recurrence vs paper's stable recurrence ---")
    print(f"{'ens':>8} {'n':>4} {'K':>3} {'seed':>4} {'v3-rec32':>12} " f"{'stable-rec32':>13}")
    for ens in ["normal", "hilbert"]:
        for n in [32, 64, 96, 128]:
            for depth in ["full", "half"]:
                N, M = n, n // 2
                K = M if depth == "full" else M // 2
                for seed in [0, 1, 2, 3, 4]:
                    rng = np.random.default_rng(seed)
                    if ens == "normal":
                        A64 = rng.standard_normal((N, M))
                    else:
                        ii = np.arange(N)[:, None]
                        jj = np.arange(M)[None, :]
                        A64 = 1.0 / (ii + jj + 1.0) + 1e-3 * rng.standard_normal((N, M))
                    v64 = rng.standard_normal(M)
                    gx = make_cotangent(N, M, K, seed + 1, np.float64)
                    x64 = gkb_forward(A64, v64, K, np.float64)
                    ref = stable_adjoint(x64, dict(A=A64, v=v64), gx, reproject=True, dtype=np.float64)

                    def err(mm):
                        return float(
                            np.sqrt(
                                (
                                    np.linalg.norm(np.asarray(mm["grad_A"], np.float64) - ref["grad_A"]) ** 2
                                    + np.linalg.norm(np.asarray(mm["grad_v"], np.float64) - ref["grad_v"]) ** 2
                                )
                                / (np.linalg.norm(ref["grad_A"]) ** 2 + np.linalg.norm(ref["grad_v"]) ** 2)
                            )
                        )

                    x32 = gkb_forward(A64, v64, K, np.float32)
                    th32 = dict(A=np.asarray(A64, np.float32), v=np.asarray(v64, np.float32))
                    m_v3 = v3_adjoint_recurrence(x32, th32, cast(gx, np.float32), dtype=np.float32)
                    m_st = stable_adjoint(x32, th32, cast(gx, np.float32), reproject=True, dtype=np.float32)
                    print(f"{ens:>8} {n:>4} {K:>3} {seed:>4} " f"{err(m_v3):>12.2e} {err(m_st):>13.2e}")


if __name__ == "__main__":
    validate_float64(7, 5, 4)
    validate_float64(6, 6, 3)
    validate_float64(12, 8, 4)
    float32_sweep()
