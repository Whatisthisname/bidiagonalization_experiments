"""Gradients of a partial-depth Krylov loss on rectangular SuiteSparse matrices.

For each matrix A (nonzero values scaled to max |a_ij| = 1), we differentiate
the k-step log-det quadrature v^T log(A^T A) v with respect to the nonzero
values of A, using a sparse matrix-vector product. The start vector
v = A^T u (u standard normal) lies in the row space of A, so that rank
deficiency does not produce zero Ritz values.

Per method we record XLA's compiled memory (arguments + outputs + temporaries,
also for programs that do not fit into RAM), the wall-clock time of
value-and-gradient where the program fits, and the float32 gradient error
against the float64 bidiagonalization adjoint on the same float32 matrix.

Matrices are downloaded from https://sparse.tamu.edu into data/suitesparse.

Usage:
    python experiment_suitesparse.py              # compute, write table
    python experiment_suitesparse.py --table-only # re-write table from JSON
"""

import argparse
import io
import json
import os
import platform
import tarfile
import time
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io
import scipy.sparse
from matfree import decomp

from algo_bidiag import bidiagonalize

MATRICES = (
    ("Kemelmacher", "Kemelmacher"),
    ("Clark", "tomographic1"),
    ("Yoshiyasu", "image_interp"),
    ("Springer", "ESOC"),
    ("Rucci", "Rucci1"),
)
NUM_MATVECS = 100
TIME_MEMORY_LIMIT_BYTES = 8e9
SLOW_SECONDS = 5.0

METHODS = ("forward", "bidiag_adjoint", "bidiag_autodiff", "lanczos_gram", "arnoldi_augmented")
GRADIENT_METHODS = METHODS[1:]
LABELS = {
    "forward": "Forward pass only",
    "bidiag_adjoint": "Bidiag adjoint (ours)",
    "bidiag_autodiff": "Bidiag autodiff",
    "lanczos_gram": r"Lanczos adjoint on $A^\top A$",
    "arnoldi_augmented": "Arnoldi adjoint, augmented",
}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "suitesparse")
RESULTS_PATH = os.path.join(HERE, "results_suitesparse.json")
TABLE_PATH = os.path.join(HERE, "latex", "table_suitesparse.tex")


def load_matrix(group, name):
    """Return (rows, cols, values, shape) in row-sorted COO form, values scaled to max |a_ij| = 1."""
    cache = os.path.join(DATA_DIR, f"{name}.npz")
    if not os.path.exists(cache):
        mtx = os.path.join(DATA_DIR, name, f"{name}.mtx")
        if not os.path.exists(mtx):
            os.makedirs(DATA_DIR, exist_ok=True)
            url = f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"
            with urllib.request.urlopen(url) as response:
                tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz").extractall(DATA_DIR)
        A = scipy.sparse.coo_matrix(scipy.io.mmread(mtx)).tocsr().tocoo()
        np.savez(cache, rows=A.row, cols=A.col, values=A.data, shape=A.shape)
    z = np.load(cache)
    values = z["values"] / np.max(np.abs(z["values"]))
    return z["rows"].astype(np.int32), z["cols"].astype(np.int32), values, tuple(int(s) for s in z["shape"])


def make_loss(method, k, rows, cols, shape):
    n_rows, n_cols = shape

    def matvec(x, values):
        return jax.ops.segment_sum(values * x[cols], rows, num_segments=n_rows, indices_are_sorted=True)

    def vecmat(y, values):
        return jax.ops.segment_sum(values * y[rows], cols, num_segments=n_cols)

    def from_bidiagonal(B, length):
        _, s, Vt = jnp.linalg.svd(B)
        return length**2 * jnp.sum(Vt[:, 0] ** 2 * jnp.log(s**2))

    def f(values, v):
        length = jnp.linalg.norm(v)
        if method in ("forward", "bidiag_adjoint", "bidiag_autodiff"):
            bd = bidiagonalize(num_matvecs=k, custom_vjp=method != "bidiag_autodiff")
            return from_bidiagonal(bd(matvec, v, values).B, length)
        if method == "lanczos_gram":
            td = decomp.tridiag_sym(k, reortho="full", custom_vjp=True)
            T = td(lambda u, a: vecmat(matvec(u, a), a), v, values).J_small
            lam, W = jnp.linalg.eigh(T)
            return length**2 * jnp.sum(W[0, :] ** 2 * jnp.log(lam))
        if method == "arnoldi_augmented":

            def matvec_aug(x, a):
                return jnp.concatenate([matvec(x[n_rows:], a), vecmat(x[:n_rows], a)])

            hs = decomp.hessenberg(2 * k, reortho="full", custom_vjp=True)
            H = hs(matvec_aug, jnp.concatenate([jnp.zeros(n_rows, v.dtype), v]), values).J_small
            offdiag = (jnp.diag(H, k=1) + jnp.diag(H, k=-1)) / 2
            B = jnp.diag(offdiag[::2]) + jnp.diag(offdiag[1::2], k=1)
            return from_bidiagonal(B, length)
        raise ValueError(method)

    if method == "forward":
        return jax.jit(f)
    return jax.jit(jax.value_and_grad(f))


def compiled_bytes(fn, *args):
    mem = fn.lower(*args).compile().memory_analysis()
    return mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes


def run_timed(fn, *args):
    """Return (output, seconds): median of 3 runs after warm-up, or a single run for slow programs."""
    out = jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    times = [time.perf_counter() - t0]
    if times[0] < SLOW_SECONDS:
        for _ in range(2):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def relative_error(x, ref):
    x = np.asarray(x, np.float64)
    if not np.all(np.isfinite(x)):
        return float("nan")
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def compute_matrix(group, name, k):
    rows, cols, values, shape = load_matrix(group, name)
    values32 = values.astype(np.float32)
    u = np.random.default_rng(0).standard_normal(shape[0])
    A64 = scipy.sparse.csr_matrix((values32.astype(np.float64), (rows, cols)), shape=shape)
    v32 = (A64.T @ u).astype(np.float32)
    print(f"== {name}: {shape[0]} x {shape[1]}, nnz {len(values)}", flush=True)

    jax.config.update("jax_enable_x64", True)
    ref_fn = make_loss("bidiag_adjoint", k, rows, cols, shape)
    ref_val, ref_grad = ref_fn(jnp.asarray(values32, jnp.float64), jnp.asarray(v32, jnp.float64))
    ref_val, ref_grad = float(ref_val), np.asarray(ref_grad)
    del ref_fn
    jax.config.update("jax_enable_x64", False)

    records = []
    args = (jnp.asarray(values32), jnp.asarray(v32))
    for m in METHODS:
        fn = make_loss(m, k, rows, cols, shape)
        nbytes = compiled_bytes(fn, *args)
        rec = {"matrix": name, "method": m, "bytes": nbytes, "seconds": None, "value_err": None, "grad_err": None}
        if nbytes < TIME_MEMORY_LIMIT_BYTES:
            out, rec["seconds"] = run_timed(fn, *args)
            val = float(out if m == "forward" else out[0])
            rec["value_err"] = abs(val - ref_val) / abs(ref_val) if np.isfinite(val) else float("nan")
            if m != "forward":
                rec["grad_err"] = relative_error(out[1], ref_grad)
        records.append(rec)
        t = f"{rec['seconds']:8.2f} s" if rec["seconds"] is not None else "       - "
        e = f"{rec['grad_err']:.1e}" if rec["grad_err"] is not None else "-"
        print(f"  {m:18s} {nbytes / 1e9:9.3f} GB {t}  grad err {e}", flush=True)
        del fn
    info = {"matrix": name, "group": group, "rows": shape[0], "cols": shape[1], "nnz": int(len(values))}
    return info, records


def compute(fresh):
    """Compute all matrices, saving after each one and skipping matrices already in the results file."""
    config = {
        "num_matvecs": NUM_MATVECS,
        "dtype": "float32",
        "device": f"{jax.devices()[0].platform}: {platform.processor() or platform.machine()}",
        "jax": jax.__version__,
        "time_memory_limit_bytes": TIME_MEMORY_LIMIT_BYTES,
        "reference": "float64 bidiag adjoint on the float32 matrix",
    }
    results = {"config": config, "matrices": [], "records": []}
    if not fresh and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            previous = json.load(f)
        if previous["config"] == config:
            results = previous
    for group, name in MATRICES:
        if any(m["matrix"] == name for m in results["matrices"]):
            print(f"== {name}: already computed", flush=True)
            continue
        info, recs = compute_matrix(group, name, NUM_MATVECS)
        results["matrices"].append(info)
        results["records"] += recs
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=1)
        jax.clear_caches()
    return results


def _fmt_bytes(b):
    gb = b / 1e9
    return f"{gb:.1f}" if gb >= 10 else f"{gb:.2f}"


def _fmt_sci(x):
    if x is None:
        return "---"
    if not np.isfinite(x):
        return "NaN"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\cdot 10^{{{int(exponent)}}}$"


def write_table(results):
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Matrix & Method & Memory (GB) & Time (s) & Gradient error \\",
        r"\midrule",
    ]
    for i, info in enumerate(results["matrices"]):
        recs = {r["method"]: r for r in results["records"] if r["matrix"] == info["matrix"]}
        header = [
            rf"\texttt{{{info['matrix'].replace('_', r'\_')}}}",
            rf"${info['rows']:,} \times {info['cols']:,}$".replace(",", "{,}"),
            rf"{info['nnz'] / 1e6:.2f}M nonzeros",
        ]
        for j, m in enumerate(METHODS):
            r = recs[m]
            seconds = f"{r['seconds']:.2f}" if r["seconds"] is not None else "---"
            error = "" if m == "forward" else _fmt_sci(r["grad_err"])
            label = rf"\textbf{{{LABELS[m]}}}" if m == "bidiag_adjoint" else LABELS[m]
            first = header[j] if j < len(header) else ""
            lines.append(f"{first} & {label} & {_fmt_bytes(r['bytes'])} & {seconds} & {error} \\\\")
        if i < len(results["matrices"]) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(TABLE_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {TABLE_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore matrices already in the results file")
    args = parser.parse_args()

    if args.table_only:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = compute(args.fresh)
    write_table(results)


if __name__ == "__main__":
    main()
