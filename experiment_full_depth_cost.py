"""Compile time and run time of the forward pass, the adjoint, and autodiff.

Two sweeps, both with the reorthogonalized forward pass and a random linear
functional of all outputs (L, R, B, residual, c) as the loss:

- compile: diagonal 950 x 950 operator (matvec v * d), against depth k;
- run time: dense random normal n x n matrices at full depth k = n, against n.

The backward pass is the VJP of the bidiagonalization, compiled separately from
the forward pass. Autodiff is skipped where XLA's compiled buffer assignment
for value-and-gradient exceeds MEMORY_LIMIT_BYTES.

Results go to results/full_depth_cost.json; `plot_figures.py` draws them.

Usage:
    python experiment_full_depth_cost.py
"""

import json
import os
import platform
import time

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from algo_bidiag import bidiagonalize

COMPILE_SIZE = 950
COMPILE_DEPTHS = (20, 40, 60, 80, 100)
RUN_SIZES = (20, 70, 120, 170, 220, 270, 320, 370, 470, 570, 670, 770, 870, 970, 1070, 1170)
REPEATS = 5
MEMORY_LIMIT_BYTES = 8e9

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "full_depth_cost.json")


def make_problem(matrix, n):
    key_a, key_v = jax.random.split(jax.random.PRNGKey(0))
    v = jax.random.normal(key_v, (n,))
    if matrix == "diagonal":
        return (lambda u, d: u * d), jax.random.normal(key_a, (n,)), v
    return (lambda u, a: a @ u), jax.random.normal(key_a, (n, n)), v


def compiled_bytes(compiled):
    mem = compiled.memory_analysis()
    return mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes


def timed_compile(fn, *args):
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    return compiled, time.perf_counter() - t0


def median_seconds(fn, *args):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def measure(matrix, n, k, custom_vjp):
    """Compile and run times of the forward pass and of its VJP."""
    matvec, params, v = make_problem(matrix, n)
    bd = bidiagonalize(num_matvecs=k, custom_vjp=custom_vjp)

    def forward(p):
        return bd(matvec, v, p)

    def forward_flat(p):
        return jax.flatten_util.ravel_pytree(forward(p))[0]

    out = jax.eval_shape(forward_flat, params)
    cotangent_flat = jax.random.normal(jax.random.PRNGKey(1), out.shape, out.dtype)

    def loss(p):
        return jnp.vdot(forward_flat(p), cotangent_flat)

    nbytes = compiled_bytes(jax.jit(jax.value_and_grad(loss)).lower(params).compile())
    record = {"matrix": matrix, "n": n, "depth": k, "custom_vjp": custom_vjp, "bytes": nbytes}
    if nbytes > MEMORY_LIMIT_BYTES:
        return record

    fwd, record["fwd_compile_s"] = timed_compile(forward, params)
    record["fwd_run_s"] = median_seconds(fwd, params)

    out, vjp = jax.vjp(forward, params)
    cotangent = jax.flatten_util.ravel_pytree(out)[1](cotangent_flat)
    bwd, record["bwd_compile_s"] = timed_compile(lambda f, c: f(c), vjp, cotangent)
    record["bwd_run_s"] = median_seconds(bwd, vjp, cotangent)
    return record


def log(record):
    keys = ("fwd_compile_s", "fwd_run_s", "bwd_compile_s", "bwd_run_s")
    timing = "  ".join(f"{key}={record[key]:.3f}" if key in record else f"{key}=-" for key in keys)
    method = "adjoint " if record["custom_vjp"] else "autodiff"
    print(f"{record['matrix']:8s} n={record['n']:5d} k={record['depth']:5d} {method} "
          f"{record['bytes'] / 1e9:7.2f} GB  {timing}", flush=True)


def main():
    records = []
    for k in COMPILE_DEPTHS:
        for custom_vjp in (True, False):
            records.append(dict(measure("diagonal", COMPILE_SIZE, k, custom_vjp), sweep="compile"))
            log(records[-1])
    for n in RUN_SIZES:
        for custom_vjp in (True, False):
            records.append(dict(measure("normal", n, n, custom_vjp), sweep="run"))
            log(records[-1])
        jax.clear_caches()

    results = {
        "config": {
            "dtype": "float32",
            "device": f"{jax.devices()[0].platform}: {platform.processor() or platform.machine()}",
            "jax": jax.__version__,
            "repeats": REPEATS,
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "compile_time": "wall-clock time of jax.jit(f).lower(...).compile()",
            "run_time": "median over repeats after a warm-up run",
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
