#!/usr/bin/env python
"""exp15: what sets the reduced-adjoint multiplier growth rate?

Hypothesis A (naive transpose argument): the exact reduced-gauge multiplier
grows backward at the same exponential rate at which UN-reorthogonalized
forward GKB loses orthogonality forward.

Hypothesis B (loop gain): the rate is governed by the per-step gain of the
backward two-term recursion,  rate <= mean_n log10( |A|_2^2 / (alpha_n
beta_n) )  (one division by alpha and one by beta per full step, with two
A-multiplications in between).

RESULTS:
  - normal ensemble: rate_adj = 0.49..0.68, rate_fwd = 0.63..1.10; ratio
    0.5-0.9 (comparable, drifting toward ~0.85 as n grows). Loop-gain proxy
    0.87-0.97: a correct upper bound, ~0.3 digits/step slack (operator
    norms overestimate the gain along the actual trajectory).
  - hilbert ensemble: HYPOTHESIS A REFUTED. rate_adj = 4.85..5.10 digits
    per step (multiplier reaches 2.6e70 at (32,16,16) -- REAL, confirmed by
    the mpmath norm check below, float64 still columnwise accurate to
    3e-14), while rate_fwd = 0.30..0.44. The loop-gain proxy 4.77-4.93
    matches rate_adj almost exactly: with alpha, beta at the noise floor
    ~3e-3 and |A| ~ 2, each backward step multiplies by ~1e5. The division
    proxy alone (mean log10 1/|beta|) = 2.2 explains under half; alpha
    divisions + A-couplings supply the rest. (exp10's "1/|beta| negligible"
    was a normal-ensemble statement; for hilbert the divisions are the
    story.)

CONCLUSION: the reduced multiplier's growth rate is the backward LOOP GAIN
|A|^2/(alpha_n beta_n) per step -- tight for ill-conditioned problems,
upper bound with modest slack for well-conditioned ones. It is NOT in
general the forward drift rate; forward drift is capped by
saturation/convergence effects that do not bound the adjoint multiplier.
The stable/V3 formulations are immune not because they damp this gain but
because their multiplier REPRESENTATION never has to hold the amplified
components (NOTES_06 sec. 2).
"""

import numpy as np

from exp02_reduced_adjoint_recurrence import solve_reduced_adjoint_recurrence
from exp05_float32_recurrence_stability import gkb_forward, make_cotangent


def forward_drift_rate(A, v, K):
    x = gkb_forward(A, v, K, np.float64, n_reortho=0)
    L, R = x["L"], x["R"]
    drift = np.zeros(K)
    for n in range(1, K):
        dl = np.abs(L[:, :n].T @ L[:, n]).max()
        dr = np.abs(R[:, :n].T @ R[:, n]).max()
        drift[n] = max(dl, dr)
    ns = np.arange(K)
    m = (drift > 1e-14) & (drift < 1e-3)
    if m.sum() < 4:
        return np.nan, drift
    return np.polyfit(ns[m], np.log10(drift[m]), 1)[0], drift


def adjoint_growth_rate(A, v, K, seed):
    N, M = A.shape
    x = gkb_forward(A, v, K, np.float64, n_reortho=2)
    gx = make_cotangent(N, M, K, seed + 1, np.float64)
    m = solve_reduced_adjoint_recurrence(x, dict(A=A, v=v), gx)
    nrm = np.linalg.norm(m["lam"], axis=0)
    ns = np.arange(1, K + 1)
    return -np.polyfit(ns, np.log10(nrm), 1)[0]


if __name__ == "__main__":
    print(f"{'ens':>8} {'n':>4} {'K':>3} {'seed':>4} {'rate_fwd':>9} " f"{'rate_adj':>9} {'ratio':>6}")
    ratios = []
    for ens in ["normal", "hilbert"]:
        for n in [32, 48, 64]:
            N, M = n, n // 2
            K = M
            for seed in [0, 1, 2, 3, 4]:
                rng = np.random.default_rng(seed)
                if ens == "normal":
                    A = rng.standard_normal((N, M))
                else:
                    ii = np.arange(N)[:, None]
                    jj = np.arange(M)[None, :]
                    A = 1.0 / (ii + jj + 1.0) + 1e-3 * rng.standard_normal((N, M))
                v = rng.standard_normal(M)
                rf, _ = forward_drift_rate(A, v, K)
                ra = adjoint_growth_rate(A, v, K, seed)
                ratio = ra / rf if np.isfinite(rf) and rf > 0 else np.nan
                if np.isfinite(ratio):
                    ratios.append(ratio)
                print(f"{ens:>8} {n:>4} {K:>3} {seed:>4} {rf:>9.3f} " f"{ra:>9.3f} {ratio:>6.2f}")
    ratios = np.array(ratios)
    print(
        f"\nratio rate_adj / rate_fwd over {len(ratios)} problems: "
        f"mean {ratios.mean():.2f}, std {ratios.std():.2f}, "
        f"range [{ratios.min():.2f}, {ratios.max():.2f}]"
    )

    # -- loop-gain proxy --------------------------------------------------
    print("\n--- loop-gain proxy  mean_n log10(|A|_2^2 / (alpha_n beta_n)) ---")
    for ens in ["normal", "hilbert"]:
        for N, M, K in [(32, 16, 16), (64, 32, 32)]:
            for seed in [0, 1, 2]:
                rng = np.random.default_rng(seed)
                if ens == "normal":
                    A = rng.standard_normal((N, M))
                else:
                    ii = np.arange(N)[:, None]
                    jj = np.arange(M)[None, :]
                    A = 1.0 / (ii + jj + 1.0) + 1e-3 * rng.standard_normal((N, M))
                v = rng.standard_normal(M)
                x = gkb_forward(A, v, K, np.float64)
                al, be = x["alphas"], x["betas"]
                nrmA = np.linalg.norm(A, 2)
                proxy = np.mean(np.log10(nrmA**2 / (al[1:] * be)))
                print(
                    f"{ens:>8} n={N:3d} K={K:2d} seed={seed}  "
                    f"loop-gain proxy {proxy:6.2f} digits/step  "
                    f"(|A|_2 = {nrmA:.2f})"
                )

    # -- attribution + reality check for the extreme hilbert rates ----------
    print("\n--- hilbert attribution: division proxy + mpmath norm check ---")
    from exp13_reduced_multiplier_is_intrinsically_huge import reduced_recurrence_mp, mp_norm, to_mp

    N, M, K = 32, 16, 16
    rng = np.random.default_rng(0)
    ii = np.arange(N)[:, None]
    jj = np.arange(M)[None, :]
    A = 1.0 / (ii + jj + 1.0) + 1e-3 * rng.standard_normal((N, M))
    v = rng.standard_normal(M)
    x = gkb_forward(A, v, K, np.float64, n_reortho=2)
    print(f"hilbert betas: {np.array2string(x['betas'], precision=1)}")
    proxy = np.mean(np.log10(1.0 / np.abs(x["betas"])))
    print(f"mean log10(1/|beta_n|) = {proxy:.2f} digits/step " f"(division-driven part of the backward growth)")
    gx = make_cotangent(N, M, K, 1, np.float64)
    m64 = solve_reduced_adjoint_recurrence(x, dict(A=A, v=v), gx)
    mmp = reduced_recurrence_mp(x, dict(A=A, v=v), gx)
    print(f"{'n':>3} {'|lam_n| mp':>12} {'|lam_n| f64':>12} {'relerr':>9}")
    for n in [0, 4, 8, 12, 15]:
        t = float(mp_norm(mmp["lam"][:, n]))
        d = float(mp_norm(mmp["lam"][:, n] - to_mp(m64["lam"][:, n])))
        print(f"{n + 1:>3} {t:>12.2e} {np.linalg.norm(m64['lam'][:, n]):>12.2e} " f"{d / t:>9.1e}")
