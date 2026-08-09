#!/usr/bin/env python
"""Probe: validate the custom bidiagonalization adjoint against unrolled autodiff.

The paper's accuracy figures only test the Jacobian of A -> L B R^T w.r.t. A.
That never exercises the cotangents of `res` and `c`, nor the gradient w.r.t.
the start vector v. This probe checks gradient agreement between the custom
adjoint (with backward reprojection) and plain reverse-mode autodiff through
the reorthogonalized forward pass, in float64, for:

  * losses touching EVERY output (L, R, alpha, beta, res, c),
  * gradients w.r.t. BOTH A and v,
  * tall / square / wide shapes, truncated and full Krylov depth k,
  * a nearly rank-deficient matrix (near-breakdown, beta ~ 1e-8).

It also prints the condition numbers of the section-6 test matrices and a
small 1x-vs-2x backward-reprojection comparison (float32).
"""

import jax
import jax.flatten_util
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from algo_bidiag import bidiagonalize


def make_losses(result_like):
    flat_like, _ = jax.flatten_util.ravel_pytree(result_like)
    w1 = jax.random.normal(jax.random.PRNGKey(7), flat_like.shape)
    w2 = jax.random.normal(jax.random.PRNGKey(8), flat_like.shape)

    def linear_loss(result):
        f, _ = jax.flatten_util.ravel_pytree(result)
        return jnp.sum(w1 * f)

    def nonlinear_loss(result):
        f, _ = jax.flatten_util.ravel_pytree(result)
        return jnp.sum(jnp.sin(w2 * f) + 0.5 * f**2)

    return {"linear(all outputs)": linear_loss, "nonlinear(all outputs)": nonlinear_loss}


def gradient_pair(A, v, k, loss_fn, *, custom: bool):
    bd = bidiagonalize(
        num_matvecs=k,
        custom_vjp=custom,
        reorthogonalize=True,
        also_reorthogonalize_vjp=True,
        reproj_repeats=1,
    )

    def objective(v_, A_):
        return loss_fn(bd(lambda vec, p: p @ vec, v_, A_))

    return jax.grad(objective, argnums=(0, 1))(v, A)


def rel_err(x, y):
    denom = jnp.maximum(jnp.linalg.norm(y), 1e-300)
    return float(jnp.linalg.norm(x - y) / denom)


def finite_difference_check(A, v, k, loss_fn, seed=3, h=1e-6):
    """Directional derivative via central differences vs the custom adjoint."""
    bd = bidiagonalize(num_matvecs=k, custom_vjp=True, reorthogonalize=True, also_reorthogonalize_vjp=True)

    def objective(v_, A_):
        return loss_fn(bd(lambda vec, p: p @ vec, v_, A_))

    kv, kA = jax.random.split(jax.random.PRNGKey(seed))
    dv = jax.random.normal(kv, v.shape)
    dA = jax.random.normal(kA, A.shape)
    plus = objective(v + h * dv, A + h * dA)
    minus = objective(v - h * dv, A - h * dA)
    fd = (plus - minus) / (2 * h)
    gv, gA = jax.grad(objective, argnums=(0, 1))(v, A)
    ad = jnp.sum(gv * dv) + jnp.sum(gA * dA)
    return float(fd), float(ad), abs(float(fd - ad) / max(abs(float(fd)), 1e-300))


def main():
    print("=" * 78)
    print("1) Adjoint (custom VJP, 1x reproj) vs unrolled autodiff, float64")
    print("=" * 78)
    cases = [
        ("tall  full", 7, 5, 5),
        ("tall  trunc", 7, 5, 3),
        ("square full", 6, 6, 6),
        ("square trunc", 12, 12, 7),
        ("wide  full", 5, 8, 5),
        ("wide  trunc", 5, 8, 3),
        ("k=1", 4, 3, 1),
    ]
    header = f"{'case':>14} {'n':>3} {'m':>3} {'k':>3} {'loss':>24} {'relerr grad_A':>15} {'relerr grad_v':>15}"
    print(header)
    worst = 0.0
    for name, n, m, k in cases:
        kA, kv = jax.random.split(jax.random.PRNGKey(hash(name) % 2**31))
        A = jax.random.normal(kA, (n, m))
        v = jax.random.normal(kv, (m,))
        bd = bidiagonalize(num_matvecs=k, custom_vjp=False, reorthogonalize=True)
        result = bd(lambda vec, p: p @ vec, v, A)
        for loss_name, loss_fn in make_losses(result).items():
            gv_c, gA_c = gradient_pair(A, v, k, loss_fn, custom=True)
            gv_a, gA_a = gradient_pair(A, v, k, loss_fn, custom=False)
            eA, ev = rel_err(gA_c, gA_a), rel_err(gv_c, gv_a)
            worst = max(worst, eA, ev)
            print(f"{name:>14} {n:>3} {m:>3} {k:>3} {loss_name:>24} {eA:>15.3e} {ev:>15.3e}")
    print(f"\nWorst relative error overall: {worst:.3e}")

    print()
    print("Finite-difference anchor (tall full case, linear loss):")
    kA, kv = jax.random.split(jax.random.PRNGKey(1234))
    A = jax.random.normal(kA, (7, 5))
    v = jax.random.normal(kv, (5,))
    bd = bidiagonalize(num_matvecs=5, custom_vjp=False, reorthogonalize=True)
    loss_fn = make_losses(bd(lambda vec, p: p @ vec, v, A))["linear(all outputs)"]
    fd, ad, err = finite_difference_check(A, v, 5, loss_fn)
    print(f"  central-diff: {fd:+.10e}   adjoint: {ad:+.10e}   rel err: {err:.3e}")

    print()
    print("=" * 78)
    print("2) Near-breakdown: A = rank-2 + 1e-8 noise, (n,m,k)=(7,5,4)")
    print("=" * 78)
    kU, kV, kN, kv = jax.random.split(jax.random.PRNGKey(99), 4)
    U = jax.random.normal(kU, (7, 2))
    Vt = jax.random.normal(kV, (2, 5))
    A = U @ Vt + 1e-8 * jax.random.normal(kN, (7, 5))
    v = jax.random.normal(kv, (5,))
    bd = bidiagonalize(num_matvecs=4, custom_vjp=False, reorthogonalize=True)
    result = bd(lambda vec, p: p @ vec, v, A)
    print("  alphas:", result.as_, "\n  betas: ", result.bs)
    loss_fn = make_losses(result)["linear(all outputs)"]
    gv_c, gA_c = gradient_pair(A, v, 4, loss_fn, custom=True)
    gv_a, gA_a = gradient_pair(A, v, 4, loss_fn, custom=False)
    print(f"  |grad_A| adjoint: {jnp.linalg.norm(gA_c):.3e}  autodiff: {jnp.linalg.norm(gA_a):.3e}")
    print(f"  rel err grad_A: {rel_err(gA_c, gA_a):.3e}   grad_v: {rel_err(gv_c, gv_a):.3e}")

    print()
    print("=" * 78)
    print("3) Section-6 matrix claims")
    print("=" * 78)
    idx = jnp.arange(40)
    C = 1 / (1 + jnp.abs(idx[:, None] - idx[None, :]))
    eigs = jnp.linalg.eigvalsh(C)
    print(
        f"  'Cauchy' C_ij=1/(1+|i-j|), n=40: cond = {eigs[-1]/eigs[0]:.4g}, min eig = {eigs[0]:.4g} (PD: {bool(eigs[0] > 0)})"
    )
    for n in [8, 12, 20, 40]:
        a = jnp.arange(n)
        H = 1 / (1 + a[:, None] + a[None, :])
        print(f"  Hilbert-like 1/(1+i+j), n={n}: cond = {jnp.linalg.cond(H):.4g}")

    print()
    print("=" * 78)
    print("4) Backward reprojection 1x vs 2x (float32, identity-Jacobian metric)")
    print("=" * 78)
    jax.config.update("jax_enable_x64", False)

    def identity_jac_err(n, repeats):
        m = n // 2
        A = jax.random.normal(jax.random.PRNGKey(1), (n, m)).astype(jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(2), (m,)).astype(jnp.float32)
        flat, unflatten = jax.flatten_util.ravel_pytree(A)
        bd = bidiagonalize(
            num_matvecs=m, custom_vjp=True, reorthogonalize=True, also_reorthogonalize_vjp=True, reproj_repeats=repeats
        )

        @jax.jit
        def reconstruct(x):
            a = unflatten(x)
            r = bd(lambda vec, p: p @ vec, v, a)
            B = jnp.diag(r.as_) + jnp.diag(r.bs, k=1)
            return jax.flatten_util.ravel_pytree(r.ls @ B @ r.rs.T)[0]

        eye = jnp.eye(len(flat), dtype=jnp.float32)
        diff = eye - jax.jacrev(reconstruct)(flat)
        return float(jnp.sqrt(jnp.mean(diff**2)))

    for n in [32, 64, 96]:
        e1 = identity_jac_err(n, 1)
        e2 = identity_jac_err(n, 2)
        print(f"  n={n:3d} (k=n/2): 1x reproj: {e1:.4e}   2x reproj: {e2:.4e}")


if __name__ == "__main__":
    main()
