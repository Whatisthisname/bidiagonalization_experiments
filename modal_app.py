"""Run experiments on a Modal GPU.

The image installs the CUDA build of the same JAX and matfree versions as
requirements.txt and ships the local modules the experiments import.

Usage:
    modal run modal_app.py    # smoke test: partial-depth cost on one GPU
"""

import json
import os

import modal

GPU = "L4"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda12]==0.11.2", "matfree==0.6.2", "numpy")
    .env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})
    .add_local_python_source("algo_bidiag", "experiment_partial_depth_cost")
)
app = modal.App("bidiag-experiments", image=image)


def compiled_bytes(fn, *args):
    # Autotuning runs candidate kernels during compilation, which fails for
    # programs that do not fit on the device.
    compiled = fn.lower(*args).compile(compiler_options={"xla_gpu_autotune_level": 0})
    mem = compiled.memory_analysis()
    return mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes


@app.function(gpu=GPU, timeout=30 * 60)
def partial_depth_cost(sizes, depth, methods):
    """Memory and run time of value-and-gradient for the convolution operator.

    Programs whose compiled buffers exceed 80% of the device memory are not run.
    """
    import jax

    import experiment_partial_depth_cost as exp

    device = jax.devices()[0]
    memory_limit = device.memory_stats()["bytes_limit"]
    print(f"{device.platform}: {device.device_kind}, {memory_limit / 1e9:.1f} GB usable", flush=True)
    exp.check_gradients_agree()

    records = []
    for n in sizes:
        theta, v = exp.make_inputs(n)
        for method in methods:
            fn = exp.make_loss(method, depth, n)
            nbytes = compiled_bytes(fn, theta, v)
            seconds = exp.wall_clock_seconds(fn, theta, v) if nbytes < 0.8 * memory_limit else None
            records.append({"size": n, "depth": depth, "method": method, "bytes": nbytes, "seconds": seconds})
            t = f"{seconds * 1e3:9.1f} ms" if seconds is not None else "        -   "
            print(f"n={n:>9,} k={depth} {method:16s} {nbytes / 1e9:8.2f} GB {t}", flush=True)
    return {
        "config": {"device": device.device_kind, "jax": jax.__version__, "memory_limit_bytes": memory_limit},
        "records": records,
    }


@app.local_entrypoint()
def main():
    results = partial_depth_cost.remote(
        sizes=(20_000, 1_000_000),
        depth=64,
        methods=("forward", "bidiag_adjoint", "bidiag_autodiff", "lanczos_gram"),
    )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "modal_smoke_test.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {path}")
