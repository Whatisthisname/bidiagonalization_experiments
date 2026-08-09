"""Transparent reimplementation of the paper's stable backward recurrence
(mirrors algo_bidiag.make_body_fun) that RECORDS all multipliers.

Multiplier convention as in common.py: stationarity grad_x + J^T mu = 0 with
con_full row blocks (recR->Xi, recL->Phi, init->kappa, orthL->Sigma,
orthR->Omega, orthres->eta).  With these conventions the code's `up_i` IS
phi_i and `down_i` IS xi_i, verified numerically in exp03.

dtype parameter lets exp05 run the same code in float32.
"""

import numpy as np


def stable_adjoint(x, th, gx, reproject=True, dtype=np.float64):
    """Paper's stable backward recurrence; `reproject` toggles the mid-flight
    re-enforcement of the selection identities (the paper's 'reprojection')."""
    L = np.asarray(x["L"], dtype=dtype)
    R = np.asarray(x["R"], dtype=dtype)
    al = np.asarray(x["alphas"], dtype=dtype)
    be = np.asarray(x["betas"], dtype=dtype)
    res = np.asarray(x["res"], dtype=dtype)
    c = dtype(np.asarray(x["c"]))
    A = np.asarray(th["A"], dtype=dtype)
    N, K = L.shape
    M = R.shape[0]

    dL = np.asarray(gx["L"], dtype=dtype)
    dR = np.asarray(gx["R"], dtype=dtype)
    da = np.asarray(gx["alphas"], dtype=dtype)
    db = np.asarray(gx["betas"], dtype=dtype)
    dres = np.asarray(gx["res"], dtype=dtype)
    dc = dtype(np.asarray(gx["c"]))

    bs = np.append(be, dtype(-1.0))  # divide by bs[i-1]; bs[-1] = -1
    dbs = np.append(db, dtype(0.0))
    das = np.append(da, dtype(0.0))

    eta = -R.T @ dres
    eta[K - 1] += da[K - 1]
    down = dres + R @ eta  # xi_{K-1} (0-indexed col K-1)

    Sigma = np.zeros((K, K), dtype=dtype)
    Omega = np.zeros((K, K), dtype=dtype)
    Phi = np.zeros((N, K), dtype=dtype)
    Xi = np.zeros((M, K), dtype=dtype)
    up_next = np.zeros(N, dtype=dtype)

    for i in range(K - 1, -1, -1):
        if reproject:
            mask = np.zeros(K, dtype=dtype)
            mask[: i + 2] = 1.0  # rows m <= i+1
            down = down - R @ (mask * (R.T @ down)) + R[:, i] * das[i]
        Xi[:, i] = down

        A_down = A @ down
        srow = -(L.T @ (dL[:, i] + A_down))
        if i - 1 >= 0:
            srow[i - 1] += al[i] * dbs[i - 1]
        srow[i] += bs[i] * dbs[i]  # zero at i=K-1 since dbs[K-1]=0
        m = np.zeros(K, dtype=dtype)
        m[: i + 1] = 1.0
        srow = srow * m
        Sigma[i, :] = srow
        Sigma[i, i] /= 2.0

        up = dL[:, i] + A_down + L @ (Sigma + Sigma.T)[:, i] - up_next * bs[i] * (i < K - 1)
        up /= al[i]
        if reproject:
            mask = np.zeros(K, dtype=dtype)
            mask[: i + 1] = 1.0  # rows m <= i
            up = up - L @ (mask * (L.T @ up))
            if i - 1 >= 0:
                up = up + L[:, i - 1] * dbs[i - 1]
        Phi[:, i] = up

        AT_up = A.T @ up
        orow = -(R.T @ (dR[:, i] + AT_up))
        if i - 1 >= 0:
            orow[i - 1] += das[i - 1] * bs[i - 1]
        orow[i] += al[i] * das[i] - (c * dc if i == 0 else dtype(0.0))
        m = np.zeros(K, dtype=dtype)
        m[: i + 1] = 1.0
        orow = orow * m
        Omega[i, :] = orow
        Omega[i, i] /= 2.0

        down = dR[:, i] + AT_up + R @ (Omega + Omega.T)[:, i] - al[i] * down + res * eta[i]
        down /= bs[i - 1] if i >= 1 else dtype(-1.0)
        up_next = up

    kappa = down
    grad_A = Phi @ R.T + L @ Xi.T
    grad_v = -c * kappa
    return dict(Phi=Phi, Xi=Xi, Sigma=Sigma, Omega=Omega, eta=eta, kappa=kappa, grad_A=grad_A, grad_v=grad_v)
