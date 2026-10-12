"""Where the run time of bidiagonalization and of Lanczos on A^T A goes.

Both methods fully reorthogonalize, but bidiagonalization orthogonalizes
against two bases, L (N rows) and R (M rows), whereas Lanczos on A^T A only
keeps R. For each SuiteSparse matrix of `experiment_suitesparse.py` and for the
convolution operator of `experiment_partial_depth_cost.py`, this script times
the k-step forward pass of both methods with and without reorthogonalization.
The difference is the time spent orthogonalizing.

Usage:
    python experiment_lanczos_cost.py
"""

import json
import os
import platform

import jax
import jax.numpy as jnp
import numpy as np
from matfree import decomp

import experiment_partial_depth_cost as conv_experiment
from algo_bidiag import bidiagonalize
from experiment_suitesparse import MATRICES, NUM_MATVECS, load_matrix, run_timed

CONV_DEPTH = 128

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "lanczos_cost.json")


def forward_programs(matvec, vecmat, k):
    """jit-compiled k-step forward passes, keyed by (method, reorthogonalize)."""

    def gram(u, p):
        return vecmat(matvec(u, p), p)

    def bidiag(reortho):
        bd = bidiagonalize(num_matvecs=k, reorthogonalize=reortho)
        return jax.jit(lambda p, v: bd(matvec, v, p).B)

    def lanczos(reortho):
        td = decomp.tridiag_sym(k, reortho="full" if reortho else "none", custom_vjp=True)
        return jax.jit(lambda p, v: td(gram, v, p).J_small)

    return {
        ("bidiag", True): bidiag(True),
        ("bidiag", False): bidiag(False),
        ("lanczos_gram", True): lanczos(True),
        ("lanczos_gram", False): lanczos(False),
    }


def time_programs(name, shape, nnz, matvec, vecmat, params, v, k):
    records = []
    for (method, reortho), fn in forward_programs(matvec, vecmat, k).items():
        _, seconds = run_timed(fn, params, v)
        records.append(
            {
                "operator": name,
                "rows": shape[0],
                "cols": shape[1],
                "nnz": nnz,
                "depth": k,
                "method": method,
                "reorthogonalize": reortho,
                "seconds": seconds,
            }
        )
        print(f"  {method:13s} reortho={reortho!s:5s} {seconds:8.3f} s", flush=True)
    return records


def suitesparse_records():
    records = []
    for group, name in MATRICES:
        rows, cols, values, shape = load_matrix(group, name)
        n_rows, n_cols = shape

        def matvec(x, a):
            return jax.ops.segment_sum(a * x[cols], rows, num_segments=n_rows, indices_are_sorted=True)

        def vecmat(y, a):
            return jax.ops.segment_sum(a * y[rows], cols, num_segments=n_cols)

        values32 = values.astype(np.float32)
        u = np.random.default_rng(0).standard_normal(n_rows).astype(np.float32)
        v = vecmat(jnp.asarray(u), jnp.asarray(values32))
        print(f"== {name}: {n_rows} x {n_cols}", flush=True)
        records += time_programs(name, shape, int(len(values)), matvec, vecmat, jnp.asarray(values32), v, NUM_MATVECS)
        jax.clear_caches()
    return records


def conv_records():
    n = conv_experiment.SIZE
    theta, v = conv_experiment.make_inputs(n)
    conv = conv_experiment.conv

    def vecmat(y, th):
        _, vjp = jax.vjp(lambda w: conv(w, th), jnp.zeros(n, y.dtype))
        return vjp(y)[0]

    shape = (n + conv_experiment.KERNEL_LENGTH - 1, n)
    print(f"== convolution: {shape[0]} x {shape[1]}", flush=True)
    return time_programs("convolution", shape, None, conv, vecmat, theta, v, CONV_DEPTH)


def main():
    records = conv_records() + suitesparse_records()
    results = {
        "config": {
            "dtype": "float32",
            "device": f"{jax.devices()[0].platform}: {platform.processor() or platform.machine()}",
            "jax": jax.__version__,
            "timing": "median of 3 runs after warm-up, or a single run above 5 s",
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
