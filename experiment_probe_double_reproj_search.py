#!/usr/bin/env python
"""Search for a context where TWO backward reprojections beat ONE.

Standalone probe. Does not modify the figure-producing experiments. Sweeps
matrix type, shape, dtype, and Krylov depth, measuring the identity-Jacobian
RMS error of the bidiagonalization adjoint for reproj_repeats in {1, 2, 3}.

We flag configurations where 2 repetitions improve meaningfully over 1.
"""

import itertools

import jax
import jax.flatten_util
import jax.numpy as jnp

from algo_bidiag import bidiagonalize


def hilbert_matrix(n):
    a = jnp.arange(n)
    return 1 / (1 + (a[:, None] + a[None, :]))


def normal_matrix(n, m, dtype):
    return jax.random.normal(jax.random.PRNGKey(0), shape=(n, m)).astype(dtype)


def identity_jacobian_error(A, v, *, reproj_repeats: int) -> float:
    flat, unflatten = jax.flatten_util.ravel_pytree(A)
    k = A.shape[1]
    bd = bidiagonalize(
        num_matvecs=k,
        custom_vjp=True,
        reorthogonalize=True,
        also_reorthogonalize_vjp=True,
        reproj_repeats=reproj_repeats,
    )

    @jax.jit
    def reconstruct(x):
        a = unflatten(x)
        result = bd(lambda vec, p: p @ vec, v, a)
        B = jnp.diag(result.as_) + jnp.diag(result.bs, k=1)
        return jax.flatten_util.ravel_pytree(result.ls @ B @ result.rs.T)[0]

    eye = jnp.eye(len(flat), dtype=A.dtype)
    diff = eye - jax.jacrev(reconstruct)(flat)
    return float(jnp.sqrt(jnp.mean(diff**2)))


def build(matrix_kind, n, shape, dtype):
    m = n if shape == "square" else max(1, n // 2)
    if matrix_kind == "hilbert":
        A = hilbert_matrix(n)[:, :m].astype(dtype)
    else:
        A = normal_matrix(n, m, dtype)
    v = jax.random.normal(jax.random.PRNGKey(1), shape=(m,)).astype(dtype)
    return A, v


def run_sweep(dtype, label):
    print(f"\n================ dtype = {label} ================")
    header = f"{'matrix':>8} {'shape':>9} {'n':>4} {'k':>4} " \
             f"{'1x':>13} {'2x':>13} {'3x':>13}  flag"
    print(header)
    wins = []
    for matrix_kind, shape in itertools.product(
        ["normal", "hilbert"], ["square", "rect"]
    ):
        for n in [8, 12, 16, 20, 24, 28, 32]:
            A, v = build(matrix_kind, n, shape, dtype)
            k = A.shape[1]
            errs = {}
            for r in (1, 2, 3):
                try:
                    errs[r] = identity_jacobian_error(A, v, reproj_repeats=r)
                except Exception:
                    errs[r] = float("nan")

            def fmt(x):
                return f"{x:13.4e}"

            # "win" = 2x finite and at least 2x smaller than 1x (or 1x non-finite)
            e1, e2 = errs[1], errs[2]
            flag = ""
            improved = (
                jnp.isfinite(jnp.array(e2))
                and (not jnp.isfinite(jnp.array(e1)) or e2 < 0.5 * e1)
            )
            if improved:
                flag = "  <-- 2x better"
                wins.append((matrix_kind, shape, n, k, e1, e2))
            print(
                f"{matrix_kind:>8} {shape:>9} {n:>4} {k:>4} "
                f"{fmt(errs[1])} {fmt(errs[2])} {fmt(errs[3])}{flag}"
            )
    return wins


def main():
    all_wins = {}

    prev_x64 = jax.config.read("jax_enable_x64")

    jax.config.update("jax_enable_x64", False)
    all_wins["float32"] = run_sweep(jnp.float32, "float32")

    jax.config.update("jax_enable_x64", True)
    all_wins["float64"] = run_sweep(jnp.float64, "float64")

    jax.config.update("jax_enable_x64", prev_x64)

    print("\n================ summary of 2x-better cases ================")
    any_win = False
    for label, wins in all_wins.items():
        for matrix_kind, shape, n, k, e1, e2 in wins:
            any_win = True
            print(
                f"{label}: {matrix_kind} {shape} n={n} k={k}: "
                f"1x={e1:.3e} -> 2x={e2:.3e}"
            )
    if not any_win:
        print("No configuration where 2 reprojections clearly beat 1.")


if __name__ == "__main__":
    main()
