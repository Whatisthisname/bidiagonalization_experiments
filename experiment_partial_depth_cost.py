"""Memory and wall-clock cost of gradients of a partial-depth Krylov loss.

Matrix-free operator: 1D convolution A(theta) v = conv(v, theta, "full"),
mapping R^n -> R^(n + 30), with a learnable kernel theta of length 31. The loss
is the k-step log-det quadrature v^T log(A^T A) v, differentiated with respect
to theta. Methods are the same as in `experiment_precision_sweep.py`, plus the
forward pass alone.

Memory is XLA's compiled buffer assignment (arguments + outputs + temporaries)
for jit(value_and_grad), so it is reported even where the program would not
fit into this machine's RAM. Time is measured only where the program fits.

Results go to results/partial_depth_cost.json; `plot_figures.py` draws them.

Usage:
    python experiment_partial_depth_cost.py
"""

import json
import os
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
from matfree import decomp

from algo_bidiag import bidiagonalize

SIZE = 20_000
KERNEL_LENGTH = 31
MEMORY_DEPTHS = (16, 32, 64, 128, 256, 512)
TIME_DEPTHS = (16, 32, 64, 128, 256)
TIME_MEMORY_LIMIT_BYTES = 12e9
TIME_REPEATS = 3

METHODS = ("forward", "bidiag_adjoint", "bidiag_autodiff", "lanczos_gram", "arnoldi_augmented")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "partial_depth_cost.json")


def conv(v, theta):
    return jnp.convolve(v, theta, mode="full")


def make_loss(method, k, n):
    n_rows = n + KERNEL_LENGTH - 1

    def from_bidiagonal(B, length):
        _, s, Vt = jnp.linalg.svd(B)
        return length**2 * jnp.sum(Vt[:, 0] ** 2 * jnp.log(s**2))

    def f(theta, v):
        length = jnp.linalg.norm(v)
        if method in ("forward", "bidiag_adjoint", "bidiag_autodiff"):
            bd = bidiagonalize(num_matvecs=k, custom_vjp=method != "bidiag_autodiff")
            return from_bidiagonal(bd(conv, v, theta).B, length)
        if method == "lanczos_gram":

            def gram(u, th):
                y, vjp = jax.vjp(lambda w: conv(w, th), u)
                return vjp(y)[0]

            T = decomp.tridiag_sym(k, reortho="full", custom_vjp=True)(gram, v, theta).J_small
            lam, W = jnp.linalg.eigh(T)
            return length**2 * jnp.sum(W[0, :] ** 2 * jnp.log(lam))
        if method == "arnoldi_augmented":

            def matvec_aug(x, th):
                _, vjp = jax.vjp(lambda w: conv(w, th), x[n_rows:])
                return jnp.concatenate([conv(x[n_rows:], th), vjp(x[:n_rows])[0]])

            hs = decomp.hessenberg(2 * k, reortho="full", custom_vjp=True)
            H = hs(matvec_aug, jnp.concatenate([jnp.zeros(n_rows, v.dtype), v]), theta).J_small
            offdiag = (jnp.diag(H, k=1) + jnp.diag(H, k=-1)) / 2
            B = jnp.diag(offdiag[::2]) + jnp.diag(offdiag[1::2], k=1)
            return from_bidiagonal(B, length)
        raise ValueError(method)

    if method == "forward":
        return jax.jit(f)
    return jax.jit(jax.value_and_grad(f))


def make_inputs(n):
    key_theta, key_v = jax.random.split(jax.random.PRNGKey(0))
    theta = jnp.zeros(KERNEL_LENGTH).at[KERNEL_LENGTH // 2].set(1.0)
    theta = theta + 0.1 * jax.random.normal(key_theta, (KERNEL_LENGTH,))
    v = jax.random.normal(key_v, (n,))
    return theta, v


def compiled_bytes(fn, *args):
    mem = fn.lower(*args).compile().memory_analysis()
    return mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes


def wall_clock_seconds(fn, *args):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(TIME_REPEATS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def check_gradients_agree(n=2_000, k=16):
    """Sanity check: all gradient methods agree in float32 on a small problem."""
    theta, v = make_inputs(n)
    grads = {m: make_loss(m, k, n)(theta, v)[1] for m in METHODS[1:]}
    ref = grads["bidiag_adjoint"]
    for m, g in grads.items():
        print(f"[sanity] k={k} {m}: rel. diff to bidiag adjoint {jnp.linalg.norm(g - ref) / jnp.linalg.norm(ref):.1e}")


def compute():
    check_gradients_agree()
    theta, v = make_inputs(SIZE)
    records = []
    for k in sorted(set(MEMORY_DEPTHS) | set(TIME_DEPTHS)):
        for m in METHODS:
            fn = make_loss(m, k, SIZE)
            nbytes = compiled_bytes(fn, theta, v)
            seconds = None
            if k in TIME_DEPTHS and nbytes < TIME_MEMORY_LIMIT_BYTES:
                seconds = wall_clock_seconds(fn, theta, v)
            records.append({"method": m, "depth": k, "bytes": nbytes, "seconds": seconds})
            t = f"{seconds * 1e3:9.1f} ms" if seconds is not None else "        -   "
            print(f"k={k:4d} {m:18s} {nbytes / 1e6:12.1f} MB {t}", flush=True)
    return {
        "config": {
            "size": SIZE,
            "kernel_length": KERNEL_LENGTH,
            "dtype": "float32",
            "device": f"{jax.devices()[0].platform}: {platform.processor() or platform.machine()}",
            "jax": jax.__version__,
            "time_repeats": TIME_REPEATS,
            "time_memory_limit_bytes": TIME_MEMORY_LIMIT_BYTES,
        },
        "records": records,
    }


def main():
    results = compute()
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
