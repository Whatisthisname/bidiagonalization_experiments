#!/usr/bin/env python
"""Compare adjoint vs autodiff on Hilbert (and normal) matrices.

Sweeps matrix type, shape, dtype, size, and Krylov depth. Reports
identity-Jacobian RMS error for all four bidiagonalization gradient modes.
Flags cases where the adjoint (with reprojection) beats every autodiff variant.
"""

import itertools
from dataclasses import dataclass

import jax
import jax.flatten_util
import jax.numpy as jnp

from algo_bidiag import bidiagonalize


def hilbert_matrix(n):
    a = jnp.arange(n)
    return 1 / (1 + (a[:, None] + a[None, :]))


def normal_matrix(n, m, dtype):
    return jax.random.normal(jax.random.PRNGKey(0), shape=(n, m)).astype(dtype)


@dataclass(frozen=True)
class Mode:
    name: str
    adjoint: bool
    reortho: bool
    reproj: bool


MODES = (
    Mode("autodiff w/o reortho", adjoint=False, reortho=False, reproj=False),
    Mode("autodiff w/ reortho", adjoint=False, reortho=True, reproj=False),
    Mode("adjoint w/o reproj", adjoint=True, reortho=True, reproj=False),
    Mode("adjoint w/ reproj", adjoint=True, reortho=True, reproj=True),
)


def identity_jacobian_error(A, v, *, k: int, mode: Mode) -> float:
    flat, unflatten = jax.flatten_util.ravel_pytree(A)
    bd = bidiagonalize(
        num_matvecs=k,
        custom_vjp=mode.adjoint,
        reorthogonalize=mode.reortho,
        also_reorthogonalize_vjp=mode.reproj,
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


def build(matrix_kind, n, m, dtype):
    if matrix_kind == "hilbert":
        A = hilbert_matrix(n)[:, :m].astype(dtype)
    else:
        A = normal_matrix(n, m, dtype)
    v = jax.random.normal(jax.random.PRNGKey(1), shape=(m,)).astype(dtype)
    return A, v


def best_autodiff(errs: dict[str, float]) -> tuple[str, float]:
    autodiff = {
        name: e
        for name, e in errs.items()
        if name.startswith("autodiff") and jnp.isfinite(jnp.array(e))
    }
    if not autodiff:
        return "none", float("inf")
    name = min(autodiff, key=autodiff.get)
    return name, autodiff[name]


def run_sweep(dtype, dtype_label):
    print(f"\n{'=' * 72}")
    print(f"dtype = {dtype_label}")
    print(f"{'=' * 72}")

    wins = []
    partial_wins = []

    for matrix_kind, shape in itertools.product(
        ["hilbert", "normal"], ["square", "rect"]
    ):
        print(f"\n--- {matrix_kind} / {shape} ---")
        header = (
            f"{'n':>4} {'m':>4} {'k':>4} | "
            + " ".join(f"{m.name[:12]:>12}" for m in MODES)
            + " | winner"
        )
        print(header)

        for n in [4, 6, 8, 10, 12, 16, 20, 24, 28, 32]:
            m = n if shape == "square" else max(2, n // 2)
            A, v = build(matrix_kind, n, m, dtype)

            for k in _k_values(n, m, shape):
                errs = {}
                for mode in MODES:
                    try:
                        errs[mode.name] = identity_jacobian_error(A, v, k=k, mode=mode)
                    except Exception:
                        errs[mode.name] = float("nan")

                adj_rep = errs["adjoint w/ reproj"]
                adj_no = errs["adjoint w/o reproj"]
                _, best_auto = best_autodiff(errs)

                winner = ""
                if jnp.isfinite(jnp.array(adj_rep)):
                    if not jnp.isfinite(jnp.array(best_auto)):
                        winner = "ADJ+reproj (autodiff failed)"
                        wins.append((matrix_kind, shape, n, m, k, errs, winner))
                    elif adj_rep < 0.5 * best_auto:
                        winner = f"ADJ+reproj ({adj_rep:.2e} < {best_auto:.2e})"
                        wins.append((matrix_kind, shape, n, m, k, errs, winner))
                    elif (
                        jnp.isfinite(jnp.array(adj_no))
                        and adj_no < 0.5 * best_auto
                    ):
                        winner = f"ADJ no reproj ({adj_no:.2e} < {best_auto:.2e})"
                        partial_wins.append(
                            (matrix_kind, shape, n, m, k, errs, winner)
                        )

                row = f"{n:4d} {m:4d} {k:4d} | "
                row += " ".join(
                    f"{errs[m.name]:12.4e}" if jnp.isfinite(jnp.array(errs[m.name]))
                    else f"{'nan':>12}"
                    for m in MODES
                )
                if winner:
                    row += f" | {winner}"
                print(row)

    return wins, partial_wins


def _k_values(n, m, shape):
    max_k = min(n, m)
    if shape == "square":
        return [max_k, max(2, max_k // 2), max(2, max_k // 4)]
    return [max_k, max(2, max_k // 2)]


def main():
    prev_x64 = jax.config.read("jax_enable_x64")
    all_wins = {}

    for enable_x64, label in [(False, "float32"), (True, "float64")]:
        jax.config.update("jax_enable_x64", enable_x64)
        dtype = jnp.float64 if enable_x64 else jnp.float32
        wins, partial = run_sweep(dtype, label)
        all_wins[label] = (wins, partial)

    jax.config.update("jax_enable_x64", prev_x64)

    print("\n" + "=" * 72)
    print("SUMMARY: adjoint w/ reproj clearly beats best autodiff")
    print("=" * 72)
    any_win = False
    for label, (wins, _) in all_wins.items():
        for matrix_kind, shape, n, m, k, errs, reason in wins:
            any_win = True
            print(
                f"{label:>8} {matrix_kind:>6} {shape:>6} n={n} m={m} k={k}: {reason}"
            )
            for mode in MODES:
                e = errs[mode.name]
                print(f"           {mode.name:22s} {e:.4e}")

    if not any_win:
        print("No case where adjoint w/ reproj clearly beats autodiff.")

    print("\n" + "=" * 72)
    print("SUMMARY: adjoint w/o reproj beats autodiff (but not w/ reproj)")
    print("=" * 72)
    for label, (_, partial) in all_wins.items():
        for matrix_kind, shape, n, m, k, errs, reason in partial:
            print(
                f"{label:>8} {matrix_kind:>6} {shape:>6} n={n} m={m} k={k}: {reason}"
            )


if __name__ == "__main__":
    main()
