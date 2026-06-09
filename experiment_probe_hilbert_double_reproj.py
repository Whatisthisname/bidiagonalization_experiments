#!/usr/bin/env python
"""Probe: Hilbert-matrix identity-Jacobian error with 1 vs 2 backward reprojections.

Standalone script; does not modify the existing figure experiments.
"""

import jax
import jax.flatten_util
import jax.numpy as jnp

from algo_bidiag import bidiagonalize


def hilbert_matrix(n):
    a = jnp.arange(n)
    return 1 / (1 + (a[:, None] + a[None, :]))


def loss_of_accuracy(A, v, *, reproj: bool, reproj_repeats: int) -> float:
    flat, unflatten = jax.flatten_util.ravel_pytree(A)
    k = A.shape[1]
    bd = bidiagonalize(
        num_matvecs=k,
        custom_vjp=True,
        reorthogonalize=True,
        also_reorthogonalize_vjp=reproj,
        reproj_repeats=reproj_repeats,
    )

    @jax.jit
    def reconstruct(x):
        a = unflatten(x)
        result = bd(lambda vec, p: p @ vec, v, a)
        B = jnp.diag(result.as_) + jnp.diag(result.bs, k=1)
        return jax.flatten_util.ravel_pytree(result.ls @ B @ result.rs.T)[0]

    eye = jnp.eye(len(flat), dtype=A.dtype)
    jacobian = jax.jacrev(reconstruct)(flat)
    diff = eye - jacobian
    return float(jnp.sqrt(jnp.mean(diff**2)))


def main():
    key_v = jax.random.PRNGKey(1)
    ns = [8, 12, 16, 20, 24]

    print("Square Hilbert matrices (n x n, k=n), float64")
    print(f"{'n':>4}  {'no reproj':>14}  {'1x reproj':>14}  {'2x reproj':>14}")
    for n in ns:
        A = hilbert_matrix(n)
        v = jax.random.normal(key_v, shape=(n,))
        row = [f"{n:4d}"]
        for reproj, repeats in [(False, 1), (True, 1), (True, 2)]:
            try:
                err = loss_of_accuracy(A, v, reproj=reproj, reproj_repeats=repeats)
                row.append(f"{err:14.6e}")
            except Exception as exc:
                row.append(f"{'FAIL':>14} ({type(exc).__name__})")
        print("  ".join(row))

    print()
    print("Square Hilbert matrices (n x n, k=n), float32")
    print(f"{'n':>4}  {'no reproj':>14}  {'1x reproj':>14}  {'2x reproj':>14}")
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        for n in ns:
            A = hilbert_matrix(n).astype(jnp.float32)
            v = jax.random.normal(key_v, shape=(n,)).astype(jnp.float32)
            row = [f"{n:4d}"]
            for reproj, repeats in [(False, 1), (True, 1), (True, 2)]:
                try:
                    err = loss_of_accuracy(
                        A, v, reproj=reproj, reproj_repeats=repeats
                    )
                    row.append(f"{err:14.6e}")
                except Exception as exc:
                    row.append(f"{'FAIL':>14} ({type(exc).__name__})")
            print("  ".join(row))
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


if __name__ == "__main__":
    main()
