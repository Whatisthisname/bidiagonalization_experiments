#!/usr/bin/env python
"""exp10: quantify the exponential-in-K error growth of the reduced gauge.

For the normal ensemble (n x n/2, K = n/2), measure the relative gradient
error of the float64 AND float32 reduced-gauge recurrences vs K, and fit
log10(error) ~ a + b*K. The float64 line isolates pure gauge-induced
amplification (no primal drift: float64 primal is ~1e-15 accurate at these
sizes thanks to double reorthogonalization).

Also reports, per size, the growth proxy prod_n max(1, 1/|beta_n|) --
NOT expected to be a sharp predictor, just to show divisions alone do not
explain it.

INTERPRETATION (established afterwards by exp13, see NOTES_06 sec. 2): the
fitted 0.59 digits/step is NOT roundoff accumulating in the recurrence (the
recurrence is columnwise forward-stable to ~2e-16 relative). It is the
growth rate of the exact multiplier |mu(K)| itself; the gradient error
follows the cancellation floor eps * |mu(K)| / |grad|, which is why the
float64 and float32 curves are parallel with offset = precision ratio.
"""

import numpy as np

from exp02_reduced_adjoint_recurrence import solve_reduced_adjoint_recurrence
from exp05_float32_recurrence_stability import gkb_forward, make_cotangent, cast
from stable_recurrence import stable_adjoint


def one(n, seed):
    N, M = n, n // 2
    K = M
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, M))
    v = rng.standard_normal(M)
    gx = make_cotangent(N, M, K, seed + 1, np.float64)
    x64 = gkb_forward(A, v, K, np.float64)
    th = dict(A=A, v=v)
    ref = stable_adjoint(x64, th, gx, reproject=True, dtype=np.float64)

    def err(m):
        return float(
            np.sqrt(
                (
                    np.linalg.norm(np.asarray(m["grad_A"], np.float64) - ref["grad_A"]) ** 2
                    + np.linalg.norm(np.asarray(m["grad_v"], np.float64) - ref["grad_v"]) ** 2
                )
                / (np.linalg.norm(ref["grad_A"]) ** 2 + np.linalg.norm(ref["grad_v"]) ** 2)
            )
        )

    e64 = err(solve_reduced_adjoint_recurrence(x64, th, gx, dtype=np.float64))
    x32 = gkb_forward(A, v, K, np.float32)
    th32 = dict(A=np.asarray(A, np.float32), v=np.asarray(v, np.float32))
    e32 = err(solve_reduced_adjoint_recurrence(x32, th32, cast(gx, np.float32), dtype=np.float32))
    beta_proxy = float(np.sum(np.log10(np.maximum(1.0, 1.0 / np.abs(x64["betas"])))))
    return K, e64, e32, beta_proxy


if __name__ == "__main__":
    print(f"{'n':>4} {'K':>4} {'err64(reduced)':>15} {'err32(reduced)':>15} " f"{'log10 prod 1/|beta|':>20}")
    Ks, l64 = [], []
    for n in [16, 24, 32, 40, 48, 56, 64, 80, 96]:
        es64, es32, prx = [], [], []
        for seed in [0, 1, 2]:
            K, e64, e32, p = one(n, seed)
            es64.append(e64)
            es32.append(e32)
            prx.append(p)
        g64 = float(np.exp(np.mean(np.log(es64))))
        g32 = float(np.exp(np.mean(np.log(np.maximum(es32, 1e-300)))))
        print(f"{n:>4} {K:>4} {g64:>15.2e} {g32:>15.2e} {np.mean(prx):>20.2f}")
        if np.isfinite(np.log10(g64)):
            Ks.append(K)
            l64.append(np.log10(g64))
    Ks, l64 = np.array(Ks, float), np.array(l64)
    b, a = np.polyfit(Ks, l64, 1)
    print(
        f"\nfit: log10(err64) ~ {a:.2f} + {b:.3f} * K   "
        f"=> about {b:.2f} decimal digits lost per iteration (geometric mean)"
    )
