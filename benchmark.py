import time
import json
import os
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from arnoldi import hessenberg
from bidiag_new_adjoint import bidiagonalize, BidiagOutput


jax.config.update("jax_enable_x64", True)


def _bidiag_loss(alphas, betas, ls, rs, res, c):
    flat = jax.flatten_util.ravel_pytree((alphas, betas, ls, rs, res, c))[0]
    # Fixed rng for reproducibility across runs
    rnd = jax.random.normal(jax.random.PRNGKey(0), flat.shape)
    return jnp.sum(rnd * flat)


def _make_padded_hess_loss(n: int) -> Callable:
    def padded_hess_loss(result):
        Q = result.Q_tall
        H = result.J_small
        res = result.residual
        c = result.init_length_inv

        ls = Q[:n, 1::2]
        rs = Q[n:, 0::2]
        superdiag = jnp.diag(H, k=1)
        alphas = superdiag[::2]
        betas = superdiag[1::2]
        res_trim = res[n:]
        return _bidiag_loss(alphas, betas, ls, rs, res_trim, c)

    return padded_hess_loss


def _bidiag_materialized_loss(result: BidiagOutput):
    return _bidiag_loss(
        result.as_, result.bs, result.ls, result.rs, result.res, result.c
    )


def _build_primal_and_cotangent_hess(n: int, m: int, k: int):
    # Construct augmented symmetric matvec used in backward_equivalence
    def matvec_sym(v_aug, A):
        upper, lower = jnp.split(v_aug, [n])
        return jnp.concatenate([A @ lower, A.T @ upper])

    hess_func = hessenberg(2 * k, reortho="full", custom_vjp=True)

    def primal(v, A):
        v_aug = jnp.concatenate([jnp.zeros(n, dtype=v.dtype), v])
        return hess_func(matvec_sym, (n, m), v_aug, A)

    primal_jit = jax.jit(primal)
    padded_hess_loss = _make_padded_hess_loss(n)

    def cotangent_from_output(result):
        _, grad = jax.value_and_grad(padded_hess_loss)(result)
        return grad

    return primal_jit, cotangent_from_output


def _build_primal_and_cotangent_bidiag(k: int):
    bd_func = bidiagonalize(num_matvecs=k, reorthogonalize=True, custom_vjp=True)

    def matvec(v, A):
        return A @ v

    def primal(v, A):
        return bd_func(matvec, v, A)

    primal_jit = jax.jit(primal)

    def cotangent_from_output(result: BidiagOutput):
        _, grad = jax.value_and_grad(_bidiag_materialized_loss)(result)
        return grad

    return primal_jit, cotangent_from_output


def _build_primal_and_cotangent_bidiag_autodiff(k: int):
    bd_func = bidiagonalize(num_matvecs=k, reorthogonalize=True, custom_vjp=False)

    def matvec(v, A):
        return A @ v

    def primal(v, A):
        return bd_func(matvec, v, A)

    primal_jit = jax.jit(primal)

    def cotangent_from_output(result: BidiagOutput):
        _, grad = jax.value_and_grad(_bidiag_materialized_loss)(result)
        return grad

    return primal_jit, cotangent_from_output


def _block_until_ready_pytree(x):
    return jax.tree.map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x
    )


# New profiling helpers
def _generate_inputs(profile: dict):
    key = jax.random.PRNGKey(int(profile["seed"]))
    n = int(profile["n"])  # rows
    m = int(profile.get("m", profile["n"]))  # allow rectangular later
    matrix_type = profile.get("matrix_type", "normal")
    if matrix_type == "normal":
        A = jax.random.normal(key, shape=(n, m))
    else:
        raise NotImplementedError(f"matrix_type={matrix_type} not supported yet")
    v = jax.random.normal(jax.random.split(key)[1], shape=(m,))
    return v, A


def _build_primal_and_loss(profile: dict):
    algorithm = profile["algorithm"]
    reorthogonalize = bool(profile["reorthogonalize"])
    custom_vjp = bool(profile["custom_vjp"])
    n = int(profile["n"])  # rows
    m = int(profile.get("m", profile["n"]))
    k = int(profile["k"])  # iterations

    if algorithm == "bidiag":
        bd_func = bidiagonalize(
            num_matvecs=k, reorthogonalize=reorthogonalize, custom_vjp=custom_vjp
        )

        def matvec(v, A):
            return A @ v

        def primal(v, A):
            return bd_func(matvec, v, A)

        loss_fn = _bidiag_materialized_loss
        return jax.jit(primal), loss_fn

    if algorithm == "hess_aug":

        def matvec_sym(v_aug, A):
            upper, lower = jnp.split(v_aug, [n])
            return jnp.concatenate([A @ lower, A.T @ upper])

        re_str = "full" if reorthogonalize else "none"
        hess_func = hessenberg(2 * k, reortho=re_str, custom_vjp=custom_vjp)

        def primal(v, A):
            v_aug = jnp.concatenate([jnp.zeros(n, dtype=v.dtype), v])
            return hess_func(matvec_sym, (n, m), v_aug, A)

        loss_fn = _make_padded_hess_loss(n)
        return jax.jit(primal), loss_fn

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _measure_profile(
    profile: dict, steady_repeats: int = 5, include_true_compile: bool = False
) -> dict:
    v, A = _generate_inputs(profile)
    fwd_fn, loss_fn = _build_primal_and_loss(profile)

    # Forward timings (perf_counter)
    t0 = time.perf_counter()
    out = fwd_fn(v, A)
    _block_until_ready_pytree(out)
    fwd_first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out2 = fwd_fn(v, A)
    _block_until_ready_pytree(out2)
    fwd_second_s = time.perf_counter() - t0

    fwd_steady = []
    for _ in range(steady_repeats):
        t0 = time.perf_counter()
        x = fwd_fn(v, A)
        _block_until_ready_pytree(x)
        fwd_steady.append(time.perf_counter() - t0)

    # Backward timings
    _, bar = jax.value_and_grad(loss_fn)(out)
    _, vjp_fn = jax.vjp(lambda vv, AA: fwd_fn(vv, AA), v, A)
    vjp_fn = jax.jit(vjp_fn)

    t0 = time.perf_counter()
    b = vjp_fn(bar)
    _block_until_ready_pytree(b)
    bwd_first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    b2 = vjp_fn(bar)
    _block_until_ready_pytree(b2)
    bwd_second_s = time.perf_counter() - t0

    bwd_steady = []
    for _ in range(steady_repeats):
        t0 = time.perf_counter()
        y = vjp_fn(bar)
        _block_until_ready_pytree(y)
        bwd_steady.append(time.perf_counter() - t0)

    # Optional true compile timings
    fwd_compile_s = None
    bwd_compile_s = None
    if include_true_compile:
        try:
            t0 = time.perf_counter()
            _ = fwd_fn.lower(v, A).compile()
            fwd_compile_s = time.perf_counter() - t0
        except Exception:
            fwd_compile_s = None
        try:
            t0 = time.perf_counter()
            _ = vjp_fn.lower(bar).compile()
            bwd_compile_s = time.perf_counter() - t0
        except Exception:
            bwd_compile_s = None

    return {
        "fwd_first_s": float(fwd_first_s),
        "fwd_second_s": float(fwd_second_s),
        "fwd_steady_mean_s": float(np.mean(fwd_steady)) if fwd_steady else None,
        "fwd_steady_std_s": float(np.std(fwd_steady)) if fwd_steady else None,
        "fwd_steady_repeats": int(steady_repeats),
        "bwd_first_s": float(bwd_first_s),
        "bwd_second_s": float(bwd_second_s),
        "bwd_steady_mean_s": float(np.mean(bwd_steady)) if bwd_steady else None,
        "bwd_steady_std_s": float(np.std(bwd_steady)) if bwd_steady else None,
        "bwd_steady_repeats": int(steady_repeats),
        "fwd_compile_s": fwd_compile_s,
        "bwd_compile_s": bwd_compile_s,
    }


def _context() -> dict:
    import platform

    try:
        import jaxlib

        jaxlib_version = getattr(jaxlib, "__version__", None)
    except Exception:
        jaxlib_version = None
    return {
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "jax_version": getattr(jax, "__version__", None),
        "jaxlib_version": jaxlib_version,
        "backend": jax.default_backend(),
        "devices": [
            {"id": i, "platform": d.platform, "device_kind": str(d)}
            for i, d in enumerate(jax.devices())
        ],
        "schema_version": "1",
    }


def append_record(record: dict, path: str):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_and_record(
    profiles: list[dict],
    out_path: str,
    steady_repeats: int = 5,
    include_true_compile: bool = False,
):
    pbar = tqdm(total=len(profiles), desc="Profiling")
    ctx = _context()
    for profile in profiles:
        metrics = _measure_profile(
            profile,
            steady_repeats=steady_repeats,
            include_true_compile=include_true_compile,
        )
        record = {"profile": profile, "metrics": metrics, "context": ctx}
        append_record(record, out_path)
        pbar.set_description(
            f"{profile['algorithm']} reortho={profile['reorthogonalize']} custom={profile['custom_vjp']} n={profile['n']} k={profile['k']}"
        )
        pbar.update(1)
    pbar.close()


def _profile_to_key(profile: dict) -> tuple:
    # Canonical key for de-duplication
    n = int(profile["n"])
    m = int(profile.get("m", n))
    return (
        profile["algorithm"],
        bool(profile["reorthogonalize"]),
        bool(profile["custom_vjp"]),
        n,
        m,
        int(profile["k"]),
        profile.get("matrix_type", "normal"),
        profile.get("dtype", "float64"),
        int(profile.get("seed", 0)),
    )


def _load_existing_profile_keys(path: str) -> set[tuple]:
    keys: set[tuple] = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                prof = rec.get("profile", {})
                keys.add(_profile_to_key(prof))
            except Exception:
                # Skip malformed lines
                continue
    return keys


def filter_existing_profiles(profiles: list[dict], path: str) -> list[dict]:
    existing = _load_existing_profile_keys(path)
    filtered: list[dict] = []
    for p in profiles:
        if _profile_to_key(p) not in existing:
            filtered.append(p)
    return filtered


def _run_benchmark_for_config(size: int, k: int, repeats: int, methods: list[str]):
    n = m = size

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, shape=(n, m))
    v = jax.random.normal(jax.random.split(key)[1], shape=(m,))

    # Build functions conditionally
    build_map: dict[str, tuple[Callable, Callable]] = {}
    if "hess" in methods:
        build_map["hess"] = _build_primal_and_cotangent_hess(n, m, k)
    if "bidiag_custom" in methods:
        build_map["bidiag_custom"] = _build_primal_and_cotangent_bidiag(k)
    if "bidiag_autodiff" in methods:
        build_map["bidiag_autodiff"] = _build_primal_and_cotangent_bidiag_autodiff(k)

    # Warmup compile and compute cotangents once
    primals: dict[str, object] = {}
    bars: dict[str, object] = {}
    vjp_fns: dict[str, Callable] = {}
    for method, (primal, cotan_fn) in build_map.items():
        out = primal(v, A)
        primals[method] = out
        _block_until_ready_pytree(out)
        bar = cotan_fn(out)
        bars[method] = bar
        _, vjp_fn = jax.vjp(lambda vv, AA, p=primal: p(vv, AA), v, A)
        vjp_fns[method] = jax.jit(vjp_fn)
        _block_until_ready_pytree(vjp_fns[method](bar))

    # Measure forward times
    fwd_times: dict[str, list[float]] = {m: [] for m in build_map}
    for _ in range(repeats):
        for method, (primal, _) in build_map.items():
            t0 = time.time()
            _block_until_ready_pytree(primal(v, A))
            fwd_times[method].append(time.time() - t0)

    # Measure backward times
    bwd_times: dict[str, list[float]] = {m: [] for m in build_map}
    for _ in range(repeats):
        for method in build_map:
            t0 = time.time()
            _block_until_ready_pytree(vjp_fns[method](bars[method]))
            bwd_times[method].append(time.time() - t0)

    return {k: np.array(v) for k, v in fwd_times.items()}, {
        k: np.array(v) for k, v in bwd_times.items()
    }


def benchmark(matrix_sizes, matvec_nums, methods: list[str], repeats=5):
    # Prepare per-method containers for means/stds
    fwd_mean = {m: np.zeros((len(matrix_sizes), len(matvec_nums))) for m in methods}
    fwd_std = {m: np.zeros((len(matrix_sizes), len(matvec_nums))) for m in methods}
    bwd_mean = {m: np.zeros((len(matrix_sizes), len(matvec_nums))) for m in methods}
    bwd_std = {m: np.zeros((len(matrix_sizes), len(matvec_nums))) for m in methods}

    total = len(matrix_sizes) * len(matvec_nums)
    pbar = tqdm(total=total, desc="Benchmarking")
    for i, size in enumerate(matrix_sizes):
        for j, k in enumerate(matvec_nums):
            pbar.set_description(f"n=m={size}, k={k}")
            fwd_times, bwd_times = _run_benchmark_for_config(
                int(size), int(k), repeats, methods
            )
            for m in methods:
                fwd_mean[m][i, j] = fwd_times[m].mean()
                fwd_std[m][i, j] = fwd_times[m].std()
                bwd_mean[m][i, j] = bwd_times[m].mean()
                bwd_std[m][i, j] = bwd_times[m].std()
                pbar.update(1)
    pbar.close()
    return {
        "methods": methods,
        "fwd_mean": fwd_mean,
        "fwd_std": fwd_std,
        "bwd_mean": bwd_mean,
        "bwd_std": bwd_std,
    }


## Plotting has been removed in favor of profiling + persistence workflow.

# hess_aug, bidiag
if __name__ == "__main__":
    # Generate profiles for all combinations of parameters

    profiles = []
    for alg in ["hess_aug", "bidiag"]:
        for reorth in [True, False]:
            for custom_vjp in [True, False]:
                for n in [1000]:  # np.linspace(100, 500, 5, dtype=int):
                    for k in np.linspace(20, 200, 10, dtype=int):
                        profiles.append(
                            {
                                "algorithm": alg,
                                "reorthogonalize": reorth,
                                "custom_vjp": custom_vjp,
                                "n": int(n),
                                "k": int(k),
                                "matrix_type": "normal",
                                "dtype": "float64",
                                "seed": 0,
                            }
                        )
    out_path = "benchmarks.ndjson"
    profiles = filter_existing_profiles(profiles, out_path)
    if not profiles:
        print(
            "No new profiles to run; all requested profiles already exist in benchmarks.ndjson"
        )
    else:
        print(f"Running {len(profiles)} new profiles; skipping existing ones")
    run_and_record(
        profiles,
        out_path=out_path,
        steady_repeats=5,
        include_true_compile=False,
    )
