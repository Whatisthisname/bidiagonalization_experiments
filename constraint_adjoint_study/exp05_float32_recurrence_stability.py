#!/usr/bin/env python
"""exp05: float32 stability of the three gauge recurrences + autodiff.

All methods get the SAME task: gradient of a fixed random linear functional
of ALL outputs, for A ~ normal(n, n/2), K = n/2 (the paper's stability
setting), computed end-to-end in float32.  Error = relative L2 distance to
the float64 reference gradient (stable recurrence at the float64 primal;
cross-checked against the float64 reduced recurrence).

Methods:
  autodiff32      : jax reverse-mode through the reorthogonalized loop
  reduced32       : NOTES_02 recurrence (unique multiplier of the SQUARE
                    reduced constraint set == appendix B.1)  [gauge: zeros]
  stable32-norepr : stable gauge recurrence, no mid-flight reprojection
  stable32-repr   : stable gauge recurrence with reprojection (paper method)

Claim (NOTES_02 sec. 3, NOTES_03 sec. 4): reduced32 degrades fastest even
though it is the unique solution of a perfectly square constraint system;
stability comes from the gauge, not from squareness.
"""

import numpy as np

from exp02_reduced_adjoint_recurrence import solve_reduced_adjoint_recurrence
from stable_recurrence import stable_adjoint


def gkb_forward(A, v, K, dtype, n_reortho=2):
    """Plain numpy GKB with reorthogonalization, mirroring algo_bidiag."""
    A = np.asarray(A, dtype)
    v = np.asarray(v, dtype)
    N, M = A.shape
    L = np.zeros((N, K), dtype)
    R = np.zeros((M, K), dtype)
    al = np.zeros(K, dtype)
    be = np.zeros(K - 1, dtype)
    nv = np.linalg.norm(v).astype(dtype)
    c = dtype(1.0) / nv
    R[:, 0] = c * v
    for n in range(K):
        t = A @ R[:, n]
        if n >= 1:
            t = t - be[n - 1] * L[:, n - 1]
        for _ in range(n_reortho):
            t = t - L[:, :n] @ (L[:, :n].T @ t)
        al[n] = np.linalg.norm(t)
        L[:, n] = t / al[n]
        w = A.T @ L[:, n] - al[n] * R[:, n]
        for _ in range(n_reortho):
            w = w - R[:, : n + 1] @ (R[:, : n + 1].T @ w)
        if n < K - 1:
            be[n] = np.linalg.norm(w)
            R[:, n + 1] = w / be[n]
        else:
            res = w
    return dict(L=L, R=R, alphas=al, betas=be, res=res, c=c)


def make_cotangent(N, M, K, seed, dtype):
    rng = np.random.default_rng(seed)
    return dict(
        L=rng.standard_normal((N, K)).astype(dtype),
        R=rng.standard_normal((M, K)).astype(dtype),
        alphas=rng.standard_normal(K).astype(dtype),
        betas=rng.standard_normal(K - 1).astype(dtype),
        res=rng.standard_normal(M).astype(dtype),
        c=dtype(rng.standard_normal()),
    )


def cast(d, dtype):
    return {k: (dtype(v) if np.ndim(v) == 0 else np.asarray(v, dtype)) for k, v in d.items()}


def autodiff_child_grad(A, v, K, gx64, x64: bool):
    """jax reverse-mode via a child process; x64 toggles float64/float32.
    Child asserts its own precision, so no silent promotion is possible."""
    import subprocess
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as td:
        fin = os.path.join(td, "in.npz")
        fout = os.path.join(td, "out.npz")
        np.savez(
            fin,
            A=A,
            v=v,
            K=K,
            gL=gx64["L"],
            gR=gx64["R"],
            ga=gx64["alphas"],
            gb=gx64["betas"],
            gres=gx64["res"],
            gc=gx64["c"],
        )
        subprocess.run(
            [
                os.sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "_autodiff32_child.py"),
                fin,
                fout,
            ],
            check=True,
            env={**os.environ, "JAX_ENABLE_X64": "1" if x64 else "0"},
        )
        out = np.load(fout)
        return out["gA"], out["gv"]


def relerr_pair(gA, gv, refA, refv):
    num = (
        np.linalg.norm(np.asarray(gA, np.float64) - refA) ** 2 + np.linalg.norm(np.asarray(gv, np.float64) - refv) ** 2
    )
    den = np.linalg.norm(refA) ** 2 + np.linalg.norm(refv) ** 2
    return float(np.sqrt(num / den))


def run(n, seed=0):
    N, M = n, n // 2
    K = M
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, M))
    v = rng.standard_normal(M)
    gx = make_cotangent(N, M, K, seed + 1, np.float64)

    # reference: float64 primal + float64 stable recurrence,
    # cross-checked against float64 jax autodiff (independent code path)
    x64d = gkb_forward(A, v, K, np.float64)
    th = dict(A=A, v=v)
    ref = stable_adjoint(x64d, th, gx, reproject=True, dtype=np.float64)
    gA64, gv64 = autodiff_child_grad(A, v, K, gx, x64=True)
    xcheck = relerr_pair(ref["grad_A"], ref["grad_v"], np.asarray(gA64), np.asarray(gv64))
    red64 = solve_reduced_adjoint_recurrence(x64d, th, gx, dtype=np.float64)
    red64_err = relerr_pair(red64["grad_A"], red64["grad_v"], ref["grad_A"], ref["grad_v"])

    # float32 world
    x32 = gkb_forward(A, v, K, np.float32)
    th32 = dict(A=np.asarray(A, np.float32), v=np.asarray(v, np.float32))
    gx32 = cast(gx, np.float32)

    red32 = solve_reduced_adjoint_recurrence(x32, th32, gx32, dtype=np.float32)
    st_nr = stable_adjoint(x32, th32, gx32, reproject=False, dtype=np.float32)
    st_rp = stable_adjoint(x32, th32, gx32, reproject=True, dtype=np.float32)
    gA_ad, gv_ad = autodiff_child_grad(A, v, K, gx, x64=False)

    e = {
        "autodiff32": relerr_pair(gA_ad, gv_ad, ref["grad_A"], ref["grad_v"]),
        "reduced64": red64_err,
        "reduced32": relerr_pair(red32["grad_A"], red32["grad_v"], ref["grad_A"], ref["grad_v"]),
        "stable32-norepr": relerr_pair(st_nr["grad_A"], st_nr["grad_v"], ref["grad_A"], ref["grad_v"]),
        "stable32-repr": relerr_pair(st_rp["grad_A"], st_rp["grad_v"], ref["grad_A"], ref["grad_v"]),
    }
    print(
        f"n={n:4d} K={K:3d} | ref-vs-autodiff64 {xcheck:.1e} | " + " | ".join(f"{k} {v_:9.2e}" for k, v_ in e.items())
    )
    return e


if __name__ == "__main__":
    print(
        "relative gradient error vs float64 reference "
        "(random linear loss on all outputs, A ~ N(0,1)^{n x n/2}, K = n/2)"
    )
    for n in [16, 32, 48, 64, 96, 128]:
        run(n)
