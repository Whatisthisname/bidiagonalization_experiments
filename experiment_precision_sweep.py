"""Float32 gradient accuracy of a partial-depth log-det quadrature.

Compares four ways of differentiating the k-step stochastic Lanczos quadrature
v^T log(A^T A) v of a rectangular matrix A:

- the bidiagonalization adjoint (ours),
- autodiff through the reorthogonalized bidiagonalization,
- the Lanczos adjoint applied to A^T A (matfree's `tridiag_sym`),
- the Arnoldi adjoint applied to [[0, A], [A^T, 0]] (matfree's `hessenberg`).

In exact arithmetic all four produce the same value and gradient, so the
reference is the float64 bidiagonalization adjoint on the float32-rounded
matrix, cross-checked against the three other methods in float64.

Two sweeps: condition number kappa(A) at fixed depth, and depth k at fixed
kappa(A).

Usage:
    python experiment_precision_sweep.py              # compute and plot
    python experiment_precision_sweep.py --plot-only  # re-plot from JSON
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matfree import decomp

import plot_style
from algo_bidiag import bidiagonalize

N_ROWS, N_COLS = 1000, 500
SEEDS = tuple(range(5))
KAPPA_SWEEP_DEPTH = 250
KAPPA_SWEEP_LOG10_KAPPAS = tuple(np.arange(1.0, 6.01, 0.5))
DEPTH_SWEEP_LOG10_KAPPA = 4.0
DEPTH_SWEEP_DEPTHS = (25, 50, 100, 150, 200, 250, 300)

METHODS = ("bidiag_adjoint", "bidiag_autodiff", "lanczos_gram", "arnoldi_augmented")
LABELS = {
    "bidiag_adjoint": "Bidiag adjoint (ours)",
    "bidiag_autodiff": "Bidiag autodiff",
    "lanczos_gram": r"Lanczos adjoint on $A^\top A$",
    "arnoldi_augmented": "Arnoldi adjoint, augmented",
}
STYLES = {
    "bidiag_adjoint": {"color": plot_style.ADJOINT_COLOR, "marker": "o", "zorder": 4},
    "bidiag_autodiff": {"color": plot_style.AUTODIFF_COLOR, "marker": "s", "linestyle": "--", "zorder": 3},
    "lanczos_gram": {"color": "C3", "marker": "^"},
    "arnoldi_augmented": {"color": "C2", "marker": "v", "linestyle": ":"},
}
FAILURE_CEILING = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results_precision_sweep.json")


def make_problem(log10_kappa, seed):
    """Rectangular matrix with geometrically decaying singular values 1 ... 1/kappa."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N_ROWS, N_COLS)))
    V, _ = np.linalg.qr(rng.standard_normal((N_COLS, N_COLS)))
    s = 10.0 ** (-log10_kappa * np.arange(N_COLS) / (N_COLS - 1))
    A = ((U * s) @ V.T).astype(np.float32)
    v = rng.standard_normal(N_COLS).astype(np.float32)
    return A, v


def make_logdet(method, k):
    """Return jit(value_and_grad) of A -> v^T log(A^T A) v via k-step quadrature."""

    def from_bidiagonal(B, length):
        _, s, Vt = jnp.linalg.svd(B)
        return length**2 * jnp.sum(Vt[:, 0] ** 2 * jnp.log(s**2))

    def f(A, v):
        length = jnp.linalg.norm(v)
        if method in ("bidiag_adjoint", "bidiag_autodiff"):
            bd = bidiagonalize(num_matvecs=k, custom_vjp=method == "bidiag_adjoint")
            return from_bidiagonal(bd(lambda u, a: a @ u, v, A).B, length)
        if method == "lanczos_gram":
            td = decomp.tridiag_sym(k, reortho="full", custom_vjp=True)
            T = td(lambda u, a: a.T @ (a @ u), v, A).J_small
            lam, W = jnp.linalg.eigh(T)
            return length**2 * jnp.sum(W[0, :] ** 2 * jnp.log(lam))
        if method == "arnoldi_augmented":
            n_rows = A.shape[0]

            def matvec_aug(x, a):
                return jnp.concatenate([a @ x[n_rows:], a.T @ x[:n_rows]])

            hs = decomp.hessenberg(2 * k, reortho="full", custom_vjp=True)
            H = hs(matvec_aug, jnp.concatenate([jnp.zeros(n_rows, A.dtype), v]), A).J_small
            offdiag = (jnp.diag(H, k=1) + jnp.diag(H, k=-1)) / 2
            B = jnp.diag(offdiag[::2]) + jnp.diag(offdiag[1::2], k=1)
            return from_bidiagonal(B, length)
        raise ValueError(method)

    return jax.jit(jax.value_and_grad(f))


def relative_error(x, ref):
    x = np.asarray(x, np.float64)
    if not np.all(np.isfinite(x)):
        return float("nan")
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def cases():
    """(sweep, log10_kappa, depth) for every point of both sweeps."""
    out = [("kappa", float(lk), KAPPA_SWEEP_DEPTH) for lk in KAPPA_SWEEP_LOG10_KAPPAS]
    out += [("depth", DEPTH_SWEEP_LOG10_KAPPA, k) for k in DEPTH_SWEEP_DEPTHS]
    return out


def compute():
    problems = {(lk, s): make_problem(lk, s) for _, lk, _ in cases() for s in SEEDS}
    depths = sorted({k for _, _, k in cases()})

    jax.config.update("jax_enable_x64", True)
    refs, crosscheck = {}, []
    for k in depths:
        fns = {m: make_logdet(m, k) for m in METHODS}
        for lk in sorted({lk for _, lk, kk in cases() if kk == k}):
            for s in SEEDS:
                A, v = (jnp.asarray(a, jnp.float64) for a in problems[lk, s])
                val, grad = fns["bidiag_adjoint"](A, v)
                refs[lk, k, s] = (float(val), np.asarray(grad))
                if s == SEEDS[0]:
                    for m in METHODS[1:]:
                        err = relative_error(fns[m](A, v)[1], refs[lk, k, s][1])
                        crosscheck.append({"log10_kappa": lk, "depth": k, "method": m, "grad_err": err})
                        print(f"[float64 cross-check] kappa=1e{lk:.1f} k={k} {m}: {err:.1e}", flush=True)

    jax.config.update("jax_enable_x64", False)
    records = []
    for k in depths:
        fns = {m: make_logdet(m, k) for m in METHODS}
        for sweep, lk, _ in [c for c in cases() if c[2] == k]:
            for s in SEEDS:
                ref_val, ref_grad = refs[lk, k, s]
                A, v = (jnp.asarray(a) for a in problems[lk, s])
                for m in METHODS:
                    val, grad = fns[m](A, v)
                    val = float(val)
                    records.append(
                        {
                            "sweep": sweep,
                            "method": m,
                            "log10_kappa": lk,
                            "depth": k,
                            "seed": s,
                            "value_err": abs(val - ref_val) / abs(ref_val) if np.isfinite(val) else float("nan"),
                            "grad_err": relative_error(grad, ref_grad),
                        }
                    )
                row = " ".join(f"{r['method']}={r['grad_err']:.1e}" for r in records[-len(METHODS):])
                print(f"{sweep} sweep kappa=1e{lk:.1f} k={k} seed={s}: {row}", flush=True)

    return {
        "config": {
            "n_rows": N_ROWS,
            "n_cols": N_COLS,
            "seeds": list(SEEDS),
            "kappa_sweep_depth": KAPPA_SWEEP_DEPTH,
            "depth_sweep_log10_kappa": DEPTH_SWEEP_LOG10_KAPPA,
            "dtype": "float32",
            "loss": "v^T log(A^T A) v, k-step quadrature",
            "reference": "float64 bidiag adjoint on the float32-rounded matrix",
        },
        "float64_crosscheck": crosscheck,
        "records": records,
    }


def plot(results):
    plot_style.use_paper_style()
    records = results["records"]
    cfg = results["config"]
    panels = (
        ("kappa", "log10_kappa", r"Condition number $\kappa(A)$", rf"Depth $k = {cfg['kappa_sweep_depth']}$"),
        ("depth", "depth", r"Krylov depth $k$", rf"$\kappa(A) = 10^{{{cfg['depth_sweep_log10_kappa']:.0f}}}$"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, (sweep, key, xlabel, title) in zip(axes, panels):
        for m in METHODS:
            rows = [r for r in records if r["sweep"] == sweep and r["method"] == m]
            xs = sorted({r[key] for r in rows})
            errs = np.array([[r["grad_err"] for r in rows if r[key] == x] for x in xs])
            errs = np.minimum(np.where(np.isfinite(errs), errs, np.inf), FAILURE_CEILING)
            x = 10.0 ** np.array(xs) if sweep == "kappa" else np.array(xs)
            style = dict(STYLES[m])
            ax.plot(
                x,
                np.median(errs, axis=1),
                label=LABELS[m],
                markersize=plot_style.MARKERSIZE,
                linewidth=plot_style.LINEWIDTH,
                **style,
            )
            ax.fill_between(x, errs.min(axis=1), errs.max(axis=1), color=style["color"], alpha=0.15, linewidth=0)
        ax.axhline(FAILURE_CEILING, color="gray", linestyle=":", linewidth=0.8)
        if sweep == "kappa":
            ax.set_xscale("log")
        ax.set_title(title)
        plot_style.style_axis(ax, xlabel, "", legend=False)
    axes[0].set_ylabel("Relative gradient error")
    axes[0].text(
        10.0 ** KAPPA_SWEEP_LOG10_KAPPAS[0], FAILURE_CEILING * 0.75, "NaN or 100% error",
        fontsize="x-small", color="gray", va="top",
    )
    axes[1].legend(fontsize="x-small", loc="lower right")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        t0 = time.perf_counter()
        results = compute()
        print(f"Computed in {time.perf_counter() - t0:.0f} s")
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=1)

    out = os.path.join(HERE, "latex", "figures", "fig_precision_sweep.pdf")
    plot(results).savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
