#!/usr/bin/env python
"""exp02: the reduced (B.2) adjoint is solvable -- explicit backward recurrence.

Claims tested (see NOTES_02):
  C1  The backward recurrence of NOTES_02 sec. 2 satisfies ALL stationarity
      equations of the reduced constraint set to machine precision.
  C2  Its multiplier equals the dense-LU solution of J_red^T mu = -grad_x rho.
  C3  Its gradient (grad_A, grad_v) matches plain autodiff.
  C4  The appendix-B.3 matrix form, if Sigma/Omega are read as TRIANGULAR
      matrices, is underdetermined with null-space dimension exactly K^2 - K.
      Read as DIAGONAL matrices it is square and nonsingular.
      -> this is the likely source of "I cannot solve the system".
"""

import numpy as np
import jax
import jax.numpy as jnp

from common import (
    autodiff_reference_grad,
    con_red,
    dense_adjoint_gradient,
    dims,
    flat,
    flatteners,
    forward,
    jacobians,
    make_linear_loss,
    random_problem,
    rel_err,
)


def solve_reduced_adjoint_recurrence(x, th, gx, dtype=np.float64):
    """Backward recurrence of NOTES_02 sec. 2 (numpy, dtype-parametric).

    Returns dict of multipliers (kappa, lam, rho, sig, omg) and gradient.
    """
    L, R = np.asarray(x["L"], dtype), np.asarray(x["R"], dtype)
    al, be = np.asarray(x["alphas"], dtype), np.asarray(x["betas"], dtype)
    c = dtype(np.asarray(x["c"]))
    A = np.asarray(th["A"], dtype)
    N, K = L.shape
    M = R.shape[0]
    dL, dR = np.asarray(gx["L"], dtype), np.asarray(gx["R"], dtype)
    da, db = np.asarray(gx["alphas"], dtype), np.asarray(gx["betas"], dtype)
    dres, dc = np.asarray(gx["res"], dtype), dtype(np.asarray(gx["c"]))

    lam = np.zeros((N, K), dtype)
    rho = np.zeros((M, K), dtype)
    sig = np.zeros(K, dtype)
    omg = np.zeros(K, dtype)

    rho[:, K - 1] = -dres  # (S-res)
    for i in range(K - 1, -1, -1):  # n = i+1
        l_i, r_i = L[:, i], R[:, i]
        rho_i = rho[:, i]
        # -- l side: sig_i then lam_i --------------------------------------
        ip_ll = -da[i] - r_i @ rho_i  # l_n^T lam_n, (S-al_n)
        rhs = dL[:, i] - A @ rho_i
        if i < K - 1:
            rhs = rhs + be[i] * lam[:, i + 1]
        # project (S-l_n) onto l_n: 0 = l^T rhs + al*ip_ll + 2 sig
        sig[i] = -0.5 * (l_i @ rhs + al[i] * ip_ll)
        lam[:, i] = -(rhs + 2.0 * sig[i] * l_i) / al[i]
        # -- r side: omg_i then rho_{i-1} or kappa -------------------------
        rhs_r = dR[:, i] - A.T @ lam[:, i] + al[i] * rho_i
        if i >= 1:
            # r_n^T rho_{n-1} from (S-be_{n-1})
            ip_rrho_prev = -db[i - 1] - L[:, i - 1] @ lam[:, i]
            omg[i] = -0.5 * (r_i @ rhs_r + be[i - 1] * ip_rrho_prev)
            rho[:, i - 1] = -(rhs_r + 2.0 * omg[i] * r_i) / be[i - 1]
        else:
            # r_1^T kappa = c*dc from (S-c)
            omg[0] = -0.5 * (r_i @ rhs_r + c * dc)
            kappa = -(rhs_r + 2.0 * omg[0] * r_i)

    grad_A = -(lam @ R.T + L @ rho.T)
    grad_v = -c * kappa
    return dict(kappa=kappa, lam=lam, rho=rho, sig=sig, omg=omg, grad_A=grad_A, grad_v=grad_v)


def flatten_mu(m, N, M, K):
    """con_red rows: init (M), recl (N x K, C-order ravel), recr (M x K),
    nrmL (K), nrmR (K).  recl was built as an (N,K) array raveled C-style."""
    return np.concatenate(
        [
            m["kappa"],
            m["lam"].ravel(),  # (N,K) C-order, matches recl.ravel()
            m["rho"].ravel(),  # (M,K) C-order, matches recr.ravel()
            m["sig"],
            m["omg"],
        ]
    )


def b3_matrix(x, th, triangular, N, M, K):
    """Assemble the linear map of the B.3-style system as a matrix acting on
    (lam, rho, Sigma, Omega, kappa) -> stationarity residual coefficients.
    triangular=True: Sigma, Omega lower triangular (as one might read B.3
    right after section 4).  triangular=False: diagonal (correct reading).
    Returns the (d_U x n_unknowns) matrix of the HOMOGENEOUS part.
    """
    L, R = np.asarray(x["L"]), np.asarray(x["R"])
    al, be = np.asarray(x["alphas"]), np.asarray(x["betas"])
    c = float(x["c"])
    A, v = np.asarray(th["A"]), np.asarray(th["v"])
    d_U, _ = dims(N, M, K)

    def apply(lam, rho, Sig, Omg, kappa):
        # residuals in x-coordinate order: L, R, alphas, betas, res, c
        SL = np.zeros((N, K))
        SR = np.zeros((M, K))
        Sa = np.zeros(K)
        Sb = np.zeros(K - 1)
        for n in range(K):
            t = al[n] * lam[:, n] - A @ rho[:, n] + (L @ Sig)[:, n]
            if n < K - 1:
                t = t + be[n] * lam[:, n + 1]
            SL[:, n] = t
            u = -A.T @ lam[:, n] + al[n] * rho[:, n] + (R @ Omg)[:, n]
            if n >= 1:
                u = u + be[n - 1] * rho[:, n - 1]
            else:
                u = u + kappa
            SR[:, n] = u
            Sa[n] = L[:, n] @ lam[:, n] + R[:, n] @ rho[:, n]
            if n < K - 1:
                Sb[n] = L[:, n] @ lam[:, n + 1] + R[:, n + 1] @ rho[:, n]
        Sres = rho[:, K - 1]
        Sc = -v @ kappa
        return np.concatenate([SL.ravel(), SR.ravel(), Sa, Sb, Sres, [Sc]])

    # build by columns
    n_tri = K * (K + 1) // 2
    n_SO = n_tri if triangular else K
    n_unk = N * K + M * K + 2 * n_SO + M
    Mat = np.zeros((d_U, n_unk))
    il, jl = np.tril_indices(K)

    def unpack(z):
        o = 0
        lam = z[o : o + N * K].reshape(N, K)
        o += N * K
        rho = z[o : o + M * K].reshape(M, K)
        o += M * K
        Sig = np.zeros((K, K))
        Omg = np.zeros((K, K))
        if triangular:
            Sig[il, jl] = z[o : o + n_tri]
            o += n_tri
            Omg[il, jl] = z[o : o + n_tri]
            o += n_tri
        else:
            Sig[np.arange(K), np.arange(K)] = z[o : o + K]
            o += K
            Omg[np.arange(K), np.arange(K)] = z[o : o + K]
            o += K
        kappa = z[o : o + M]
        return lam, rho, Sig, Omg, kappa

    for j in range(n_unk):
        z = np.zeros(n_unk)
        z[j] = 1.0
        Mat[:, j] = apply(*unpack(z))
    return Mat


def run(N, M, K, seed=0):
    print(f"\n=== (N, M, K) = ({N}, {M}, {K}) ===")
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)
    loss, _ = make_linear_loss(N, M, K)
    unflat_x, _ = flatteners(N, M, K)
    gx_flat = jax.grad(lambda xf: loss(unflat_x(xf)))(flat(x))
    gx = unflat_x(gx_flat)

    m = solve_reduced_adjoint_recurrence(x, th, gx)

    # C1: stationarity residual  J_red^T mu + grad_x rho = 0
    Jx, Jth = jacobians(con_red, x, th, N, M, K)
    mu = flatten_mu(m, N, M, K)
    stat = Jx.T @ mu + np.asarray(gx_flat)
    print(
        f"C1  |J_red^T mu_rec + grad_x| / |grad_x| = "
        f"{np.linalg.norm(stat) / np.linalg.norm(np.asarray(gx_flat)):.2e}"
    )

    # C2: equals dense solution
    _, mu_dense, _, _ = dense_adjoint_gradient(con_red, x, th, gx, N, M, K)
    print(f"C2  |mu_rec - mu_dense| / |mu_dense| = {rel_err(mu, mu_dense):.2e}")

    # C3: gradient matches autodiff
    gref = autodiff_reference_grad(A, v, K, loss)
    print(
        f"C3  grad_A rel err = {rel_err(m['grad_A'], np.asarray(gref['A'])):.2e}, "
        f"grad_v rel err = {rel_err(m['grad_v'], np.asarray(gref['v'])):.2e}"
    )

    # C4: triangular vs diagonal Sigma/Omega in the B.3-style matrix form
    for tri in (False, True):
        Mat = b3_matrix(x, th, tri, N, M, K)
        s = np.linalg.svd(Mat, compute_uv=False)
        tol = s[0] * max(Mat.shape) * np.finfo(np.float64).eps
        rank = int((s > tol).sum())
        nul = Mat.shape[1] - rank
        expect = (K * K - K) if tri else 0
        print(
            f"C4  Sigma/Omega {'triangular' if tri else 'diagonal  '}: "
            f"matrix {Mat.shape}, rank {rank}, null dim {nul} "
            f"(expected {expect})"
        )


if __name__ == "__main__":
    run(7, 5, 4)
    run(6, 6, 3)
    run(5, 8, 4)
