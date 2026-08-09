#!/usr/bin/env python
"""exp16: the paper's sparsity theorem (checkerboard Gamma etc.) via parity.

Claim (NOTES_07): let S = diag(I_N, -I_M) and D = diag(s_1..s_{2K}) with
s_i = (-1)^i. Then S anticommutes with the augmented matrix (S Aa S = -Aa),
the augmented start vector is S-odd, and the Hessenberg adjoint system
(paper eqs. (adjoint_hessenberg_constraint1)-(4)) is EQUIVARIANT under

    Lambda -> -S Lambda D,  Gamma -> D Gamma D,
    lambda_0 -> -S lambda_0,  gamma -> -D gamma.

Since the augmented adjoint has a UNIQUE solution, the solution is a fixed
point of this map, which forces componentwise:

    q_i pure:            S q_i = s_i q_i             (primal, induction)
    Lambda columns pure: S lambda_i = -s_i lambda_i  (opposite block to q_i)
    Gamma checkerboard:  Gamma_{ji} = 0 if s_j s_i = -1  (i+j odd)
    gamma alternating:   gamma_i = 0 for s_i = -1... precisely gamma_i = 0
                         whenever -s_i = -1, i.e. i odd -> gamma_i free,
                         i even -> gamma_i = 0    [pattern (*,0,*,0,...)]
    lambda_0 M-block:    S lambda_0 = -lambda_0  (upper N-part zero)

i.e. the ENTIRE Theorem 4.1 of the paper, without the backward induction.

This experiment verifies, at several sizes (float64, dense):
  C1 primal parity purity + H alternation (Lanczos on the augmented matrix)
  C2 the dense augmented adjoint solve reproduces the bidiagonal gradient
     (vs the study's stable recurrence)
  C3 all sparsity claims of the theorem hold to machine precision
  C4 equivariance holds as an ALGEBRAIC identity at the solution: the
     transformed multiplier satisfies the system with residual ~ eps
     (this is the step that, combined with uniqueness, PROVES C3)
  C5 the pruned nonzero blocks equal the study's stable-gauge multipliers
     (Phi, Xi, kappa, eta) -- the pruning claim of NOTES_03 sec. 5.
"""

import numpy as np

from exp05_float32_recurrence_stability import gkb_forward, make_cotangent
from stable_recurrence import stable_adjoint


def lanczos_aug(Aa, va, K2, n_reortho=2):
    """Arnoldi/Lanczos with reorthogonalization on the augmented matrix,
    mirroring the repo's algo_hessenberg.arnoldi (full H columns)."""
    n = va.shape[0]
    Q = np.zeros((n, K2))
    H = np.zeros((K2, K2))
    c = 1.0 / np.linalg.norm(va)
    Q[:, 0] = c * va
    t = va
    for i in range(K2):
        t = Aa @ Q[:, i]
        hs = Q.T @ t
        t = t - Q @ hs
        for _ in range(n_reortho):
            t = t - Q @ (Q.T @ t)
        if i + 1 < K2:
            hn = np.linalg.norm(t)
            hs[i + 1] = hn
            Q[:, i + 1] = t / hn
        H[:, i] = hs
    return Q, H, t, c  # t = residual after 2K matvecs


def build_dense_adjoint(Aa, va, Q, H, r, c, GQ, GH, Gr, Gc):
    """Paper's Hessenberg adjoint system as a dense square linear system in
    z = [vec(Lambda), tril(Gamma), lambda_0, gamma].
    Equations: E1 (full), E2 on upper-Hessenberg support (i <= j+1),
    E3, E4.  Returns (M, rhs) with M z = rhs."""
    n, K2 = Q.shape
    iG = np.tril_indices(K2)
    nG = len(iG[0])
    nz = n * K2 + nG + n + K2

    def apply(z):
        o = 0
        Lam = z[o : o + n * K2].reshape(n, K2)
        o += n * K2
        Gam = np.zeros((K2, K2))
        Gam[iG] = z[o : o + nG]
        o += nG
        lam0 = z[o : o + n]
        o += n
        gam = z[o : o + K2]
        E1 = Aa.T @ Lam - Lam @ H.T + np.outer(lam0, np.eye(K2)[0]) + Q @ Gam + Q @ Gam.T + np.outer(r, gam)
        E2 = -(Q.T @ Lam)
        E3 = -Lam[:, K2 - 1] + Q @ gam
        E4 = -(va @ lam0)
        ii, jj = np.indices((K2, K2))
        keep = ii <= jj + 1
        return np.concatenate([E1.ravel(), E2[keep].ravel(), E3, np.array([E4])])

    m_rows = n * K2 + int((np.indices((K2, K2))[0] <= np.indices((K2, K2))[1] + 1).sum()) + n + 1
    Mat = np.zeros((m_rows, nz))
    for j in range(nz):
        z = np.zeros(nz)
        z[j] = 1.0
        Mat[:, j] = apply(z)
    ii, jj = np.indices((K2, K2))
    keep = ii <= jj + 1
    rhs = -np.concatenate([GQ.ravel(), GH[keep].ravel(), Gr, np.array([Gc])])
    return Mat, rhs, iG


def run(N, M, K, seed=0):
    print(f"\n=== (N, M, K) = ({N}, {M}, {K}) ===")
    K2 = 2 * K
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, M))
    v = rng.standard_normal(M)
    Aa = np.block([[np.zeros((N, N)), A], [A.T, np.zeros((M, M))]])
    va = np.concatenate([np.zeros(N), v])
    Q, H, r, c = lanczos_aug(Aa, va, K2)

    s = np.array([(-1.0) ** i for i in range(1, K2 + 1)])  # s_i = (-1)^i
    Sdiag = np.concatenate([np.ones(N), -np.ones(M)])

    # C1: primal parity purity
    pure = max(np.linalg.norm(Sdiag[:, None] * Q - Q * s[None, :], np.inf) / max(1e-30, np.abs(Q).max()), 0.0)
    Hpar = np.abs(H * (np.outer(s, s) > 0)).max()  # same-parity entries
    resid_par = np.linalg.norm((Sdiag * r) + r)  # r should be S-odd
    print(f"C1 primal purity {pure:.1e}; H same-parity entries {Hpar:.1e}; " f"res S-odd violation {resid_par:.1e}")

    # cotangent on the bidiagonal-world outputs, embedded via extract^T
    gx = make_cotangent(N, M, K, seed + 1, np.float64)
    GQ = np.zeros((N + M, K2))
    for nn in range(K):
        GQ[N:, 2 * nn] = gx["R"][:, nn]  # col 2n-1 (0-idx 2n-2): r_n
        GQ[:N, 2 * nn + 1] = gx["L"][:, nn]  # col 2n: l_n
    GH = np.zeros((K2, K2))
    for nn in range(K):
        GH[2 * nn, 2 * nn + 1] = gx["alphas"][nn]  # superdiag alpha_n
        if nn < K - 1:
            GH[2 * nn + 1, 2 * nn + 2] = gx["betas"][nn]  # superdiag beta_n
    Gr = np.concatenate([np.zeros(N), gx["res"]])
    Gc = float(gx["c"])

    Mat, rhs, iG = build_dense_adjoint(Aa, va, Q, H, r, c, GQ, GH, Gr, Gc)
    print(f"C2 dense system {Mat.shape}, cond = {np.linalg.cond(Mat):.1e}")
    z = np.linalg.solve(Mat, rhs)
    n = N + M
    o = 0
    Lam = z[o : o + n * K2].reshape(n, K2)
    o += n * K2
    Gam = np.zeros((K2, K2))
    Gam[iG] = z[o : o + len(iG[0])]
    o += len(iG[0])
    lam0 = z[o : o + n]
    o += n
    gam = z[o : o + K2]

    # C2: gradient match vs the study's stable recurrence
    gAa = Lam @ Q.T
    gA = gAa[:N, N:] + gAa[N:, :N].T
    gv = (-c * lam0)[N:]
    x = gkb_forward(A, v, K, np.float64)
    ref = stable_adjoint(x, dict(A=A, v=v), gx, reproject=True)
    eA = np.linalg.norm(gA - ref["grad_A"]) / np.linalg.norm(ref["grad_A"])
    ev = np.linalg.norm(gv - ref["grad_v"]) / np.linalg.norm(ref["grad_v"])
    print(f"C2 grad_A vs stable recurrence {eA:.1e}; grad_v {ev:.1e}")

    # C3: sparsity claims of Theorem 4.1
    sc = np.abs(z).max()
    lam_wrong = max(np.linalg.norm(Lam[:N, i]) if s[i] > 0 else np.linalg.norm(Lam[N:, i]) for i in range(K2))
    # lambda_i pure with sign -s_i: -s_i = -1 (i even 1-idx? s_i=+1) -> M-block
    gam_checker = np.abs(Gam * (np.outer(s, s) < 0)).max()
    gam_vec_zero = np.abs(gam[s > 0]).max()  # zero where -s_i = -1
    lam0_wrong = np.linalg.norm(lam0[:N])  # lambda_0 M-block only
    print(
        f"C3 (rel. to |mu|_inf = {sc:.1e}) lambda wrong-block "
        f"{lam_wrong / sc:.1e}; Gamma checkerboard {gam_checker / sc:.1e}; "
        f"gamma even entries {gam_vec_zero / sc:.1e}; "
        f"lambda_0 N-block {lam0_wrong / sc:.1e}"
    )

    # C4: equivariance as an algebraic identity (transformed z satisfies
    # the same system)
    Lam_t = -(Sdiag[:, None] * Lam) * s[None, :]
    Gam_t = (np.outer(s, s)) * Gam
    lam0_t = -(Sdiag * lam0)
    gam_t = -(s * gam)
    z_t = np.concatenate([Lam_t.ravel(), Gam_t[iG], lam0_t, gam_t])
    res_t = np.linalg.norm(Mat @ z_t - rhs) / np.linalg.norm(rhs)
    fixed = np.linalg.norm(z_t - z) / np.linalg.norm(z)
    print(
        f"C4 residual of TRANSFORMED multiplier {res_t:.1e} "
        f"(equivariance); |z_t - z|/|z| = {fixed:.1e} (fixed point)"
    )

    # C5: pruning onto the bidiagonal stable gauge.
    # lambda_{2n-1} is S-even (N-block; attaches to the recL-type column
    # A r_n = ...); lambda_{2n} is S-odd (M-block; recR-type column).
    Phi = Lam[:N, 0::2]  # lambda_{2n-1} N-part  <-> Phi_n
    Xi = Lam[N:, 1::2]  # lambda_{2n}   M-part  <-> Xi_n
    eP = np.linalg.norm(Phi - ref["Phi"]) / np.linalg.norm(ref["Phi"])
    eX = np.linalg.norm(Xi - ref["Xi"]) / np.linalg.norm(ref["Xi"])
    eK_ = np.linalg.norm(lam0[N:] - ref["kappa"]) / np.linalg.norm(ref["kappa"])
    eta_h = gam[0::2]  # odd entries of gamma
    eE = np.linalg.norm(eta_h - ref["eta"]) / np.linalg.norm(ref["eta"])
    print(f"C5 pruned vs stable gauge: Phi {eP:.1e}, Xi {eX:.1e}, " f"kappa {eK_:.1e}, eta {eE:.1e}")


if __name__ == "__main__":
    run(7, 5, 4)
    run(6, 6, 3)
    run(12, 8, 4, seed=3)
