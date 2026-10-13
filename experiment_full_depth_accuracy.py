"""Accuracy of bidiagonalization and its gradients at full depth.

At full depth, L B R^T reconstructs A, so the Jacobian of A -> L B R^T is the
identity. We report the RMS deviation of that Jacobian from the identity
(gradient error) and the RMS error of the reconstruction itself, for

- normal: random normal n x n matrices, k = n, float64;
- toeplitz: symmetric Toeplitz matrices C_ij = 1 / (1 + |i - j|), n x n, k = n,
  float32;
- toeplitz_reconstruction: forward reconstruction error on the Toeplitz matrices;
- reprojection: random normal n x (n/2) matrices, k = n/2, float32, adjoint
  with and without the reprojection step in the backward pass.

Results go to results/full_depth_accuracy.json; `plot_figures.py` draws them.

Usage:
    python experiment_full_depth_accuracy.py
"""

import json
import os
import platform

import jax
import jax.flatten_util
import jax.numpy as jnp
import tqdm

from algo_bidiag import bidiagonalize

GRADIENT_METHODS = ("bidiag_autodiff_noreortho", "bidiag_autodiff", "bidiag_adjoint_noreproj", "bidiag_adjoint")
EXPERIMENTS = {
    "normal": {
        "matrix": "normal",
        "ns": list(range(4, 40, 3)),
        "width": "n",
        "dtype": "float64",
        "metric": "jacobian",
        "methods": GRADIENT_METHODS,
    },
    "toeplitz": {
        "matrix": "toeplitz",
        "ns": list(range(4, 41, 2)),
        "width": "n",
        "dtype": "float32",
        "metric": "jacobian",
        "methods": GRADIENT_METHODS,
    },
    "toeplitz_reconstruction": {
        "matrix": "toeplitz",
        "ns": list(range(4, 41, 2)),
        "width": "n",
        "dtype": "float32",
        "metric": "reconstruction",
        "methods": ("forward_noreortho", "forward"),
    },
    "reprojection": {
        "matrix": "normal",
        "ns": list(range(32, 121, 8)),
        "width": "n/2",
        "dtype": "float32",
        "metric": "jacobian",
        "methods": ("bidiag_adjoint_noreproj", "bidiag_adjoint"),
    },
}
METHOD_FLAGS = {
    "forward_noreortho": {"custom_vjp": True, "reorthogonalize": False},
    "forward": {"custom_vjp": True, "reorthogonalize": True},
    "bidiag_autodiff_noreortho": {"custom_vjp": False, "reorthogonalize": False},
    "bidiag_autodiff": {"custom_vjp": False, "reorthogonalize": True},
    "bidiag_adjoint_noreproj": {"custom_vjp": True, "reorthogonalize": True, "also_reorthogonalize_vjp": False},
    "bidiag_adjoint": {"custom_vjp": True, "reorthogonalize": True, "also_reorthogonalize_vjp": True},
}

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "full_depth_accuracy.json")
KEY_A = jax.random.PRNGKey(1)
_, KEY_V = jax.random.split(jax.random.PRNGKey(1), num=2)


def make_matrix(name, n):
    if name == "normal":
        return jax.random.normal(KEY_A, shape=(n, n))
    if name == "toeplitz":
        idx = jnp.arange(n)
        return 1 / (1 + jnp.abs(idx[:, None] - idx[None, :]))
    raise ValueError(name)


def error(experiment, method, n):
    cfg = EXPERIMENTS[experiment]
    dtype = jnp.dtype(cfg["dtype"])
    width = n if cfg["width"] == "n" else n // 2
    A = make_matrix(cfg["matrix"], n)[:, :width].astype(dtype)
    v = jax.random.normal(KEY_V, shape=(width,)).astype(dtype)
    flat, unflatten = jax.flatten_util.ravel_pytree(A)
    bd = bidiagonalize(num_matvecs=width, **METHOD_FLAGS[method])

    @jax.jit
    def reconstruct(x):
        out = bd(lambda u, a: a @ u, v, unflatten(x))
        return (out.L @ out.B @ out.R.T).ravel()

    if cfg["metric"] == "reconstruction":
        diff = flat - reconstruct(flat)
    else:
        diff = jnp.eye(len(flat), dtype=dtype) - jax.jacrev(reconstruct)(flat)
    return float(jnp.sqrt(jnp.mean(diff**2)))


def compute():
    records = []
    for name, cfg in EXPERIMENTS.items():
        jax.config.update("jax_enable_x64", cfg["dtype"] == "float64")
        for method in cfg["methods"]:
            for n in tqdm.tqdm(cfg["ns"], desc=f"{name}: {method}"):
                records.append({"experiment": name, "method": method, "n": n, "error": error(name, method, n)})
        jax.config.update("jax_enable_x64", False)
    return records


def main():
    records = compute()
    experiments = {name: dict(cfg, methods=list(cfg["methods"])) for name, cfg in EXPERIMENTS.items()}
    results = {
        "config": {
            "device": f"{jax.devices()[0].platform}: {platform.processor() or platform.machine()}",
            "jax": jax.__version__,
            "experiments": experiments,
            "error": "RMS of I - d(L B R^T)/dA (jacobian) or of A - L B R^T (reconstruction)",
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
