import time
import json
import os
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from hessenberg import hessenberg
from bidiag import bidiagonalize, BidiagOutput
from benchmark_plot_utils import suite_sparse_load
import jax.experimental.sparse


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


def _generate_inputs(profile: dict):
    key = jax.random.PRNGKey(int(profile["seed"]))
    # Prefer new field `matrix`; fall back to legacy `matrix_type` for compatibility
    matrix_name = profile.get("matrix", profile.get("matrix_type", "normal"))

    if matrix_name == "normal":
        n = int(profile["n"])  # rows
        m = int(profile.get("m", profile["n"]))  # allow rectangular later
        A = jax.random.normal(key, shape=(n, m))
        v = jax.random.normal(jax.random.split(key)[1], shape=(m,))
        return v, A, lambda x: x, False, None

    # Sparse matrix case: load by name using SuiteSparse files already downloaded
    A = suite_sparse_load(matrix_name, path="./data/matrices/")
    n, m = A.shape
    # Keep profile consistent for downstream shape-using code paths
    profile["n"] = int(n)
    profile["m"] = int(m)
    v = jax.random.normal(jax.random.split(key)[1], shape=(m,)).astype(A.dtype)
    params, params_unflatten = jax.flatten_util.ravel_pytree(A.data)
    return v, params, params_unflatten, True, A


def _build_primal_and_loss(profile: dict, is_sparse, params_unflatten, M):
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

        def matvec(v, params):
            if is_sparse:
                pp = params_unflatten(params)
                matrix = jax.experimental.sparse.BCOO((pp, M.indices), shape=M.shape)
                return matrix @ v
            else:
                return params @ v

        def primal(v, params):
            return bd_func(matvec, v, params)

        loss_fn = _bidiag_materialized_loss
        return primal, loss_fn

    if algorithm == "hess_aug":

        def matvec_sym(v_aug, params):
            upper, lower = jnp.split(v_aug, [n])
            if is_sparse:
                pp = params_unflatten(params)
                A = jax.experimental.sparse.BCOO((pp, M.indices), shape=M.shape)
            else:
                A = params
            return jnp.concatenate([A @ lower, A.T @ upper])

        re_str = "full" if reorthogonalize else "none"
        hess_func = hessenberg(2 * k, reortho=re_str, custom_vjp=custom_vjp)

        def primal(v, params):
            v_aug = jnp.concatenate([jnp.zeros(n, dtype=v.dtype), v])
            return hess_func(matvec_sym, (n, m), v_aug, params)

        loss_fn = _make_padded_hess_loss(n)
        return jax.jit(primal), loss_fn

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _measure_profile(
    profile: dict, steady_repeats: int = 5, include_true_compile: bool = False
) -> dict:
    v, params, params_unflatten, is_sparse, M = _generate_inputs(profile)

    fwd_fn, loss_fn = _build_primal_and_loss(
        profile, is_sparse=is_sparse, params_unflatten=params_unflatten, M=M
    )

    fwd_fn_jit = jax.jit(fwd_fn)
    t0 = time.perf_counter()
    # Forward timings
    out = fwd_fn_jit(v, params)
    _block_until_ready_pytree(out)
    fwd_compile = time.perf_counter() - t0

    fwd_steady = []
    for _ in range(steady_repeats):
        t0 = time.perf_counter()
        x = fwd_fn_jit(v, params)
        _block_until_ready_pytree(x)
        fwd_steady.append(time.perf_counter() - t0)

    # Backward timings
    _, cotan = jax.value_and_grad(loss_fn)(out)
    _, vjp_fn = jax.vjp(fwd_fn_jit, v, params)
    vjp_fn = jax.jit(vjp_fn)
    t0 = time.perf_counter()
    b2 = vjp_fn(cotan)
    _block_until_ready_pytree(b2)
    bwd_compile = time.perf_counter() - t0

    bwd_steady = []
    for _ in range(steady_repeats):
        t0 = time.perf_counter()
        y = vjp_fn(cotan)
        _block_until_ready_pytree(y)
        bwd_steady.append(time.perf_counter() - t0)

    # Optional true compile timings
    if include_true_compile and False:
        fwd_compile_times = []
        for _ in range(steady_repeats // 2):
            t0 = time.perf_counter()
            _ = jax.jit(fwd_fn).lower(v, params).compile()
            fwd_compile_times.append(time.perf_counter() - t0)
        fwd_compile_s = float(np.mean(fwd_compile_times))

        bwd_compile_times = []
        for _ in range(steady_repeats // 2):
            t0 = time.perf_counter()
            _ = jax.jit(jax.vjp(fwd_fn, v, params)[1]).lower(cotan).compile()
            bwd_compile_times.append(time.perf_counter() - t0)
        bwd_compile_s = float(np.mean(bwd_compile_times))

    return {
        "fwd_steady_mean_s": float(np.mean(fwd_steady)),
        "fwd_steady_std_s": float(np.std(fwd_steady)),
        "fwd_steady_repeats": int(steady_repeats),
        "bwd_steady_mean_s": float(np.mean(bwd_steady)),
        "bwd_steady_std_s": float(np.std(bwd_steady)),
        "bwd_steady_repeats": int(steady_repeats),
        "fwd_compile_s": fwd_compile - float(np.mean(fwd_steady)),  # fwd_compile_s,
        "bwd_compile_s": bwd_compile - float(np.mean(bwd_steady)),  # bwd_compile_s,
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
    # Prefer new `matrix` field; fall back to legacy `matrix_type`
    matrix_name = profile.get("matrix", profile.get("matrix_type", "normal"))

    # Resolve shapes; for sparse named matrices, infer from file to ensure consistent keys
    if matrix_name == "normal":
        n = int(profile["n"])
        m = int(profile.get("m", n))
    else:
        try:
            A = suite_sparse_load(matrix_name, path="./data/matrices/")
            n, m = A.shape
        except Exception:
            # Fallback to provided values if loading fails
            n = int(profile.get("n", -1))
            m = int(profile.get("m", -1))

    return (
        profile["algorithm"],
        bool(profile["reorthogonalize"]),
        bool(profile["custom_vjp"]),
        int(n),
        int(m),
        int(profile["k"]),
        matrix_name,
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


# def _run_benchmark_for_config(size: int, k: int, repeats: int, methods: list[str]):
#     n = m = size

#     key = jax.random.PRNGKey(0)
#     A = jax.random.normal(key, shape=(n, m))
#     v = jax.random.normal(jax.random.split(key)[1], shape=(m,))

#     # Build functions conditionally
#     build_map: dict[str, tuple[Callable, Callable]] = {}
#     if "hess" in methods:
#         build_map["hess"] = _build_primal_and_cotangent_hess(n, m, k)
#     if "bidiag_custom" in methods:
#         build_map["bidiag_custom"] = _build_primal_and_cotangent_bidiag(k)
#     if "bidiag_autodiff" in methods:
#         build_map["bidiag_autodiff"] = _build_primal_and_cotangent_bidiag_autodiff(k)

#     # Warmup compile and compute cotangents once
#     primals: dict[str, object] = {}
#     bars: dict[str, object] = {}
#     vjp_fns: dict[str, Callable] = {}
#     for method, (primal, cotan_fn) in build_map.items():
#         out = primal(v, A)
#         primals[method] = out
#         _block_until_ready_pytree(out)
#         bar = cotan_fn(out)
#         bars[method] = bar
#         _, vjp_fn = jax.vjp(lambda vv, AA, p=primal: p(vv, AA), v, A)
#         vjp_fns[method] = jax.jit(vjp_fn)
#         _block_until_ready_pytree(vjp_fns[method](bar))

#     # Measure forward times
#     fwd_times: dict[str, list[float]] = {m: [] for m in build_map}
#     for _ in range(repeats):
#         for method, (primal, _) in build_map.items():
#             t0 = time.time()
#             _block_until_ready_pytree(primal(v, A))
#             fwd_times[method].append(time.time() - t0)

#     # Measure backward times
#     bwd_times: dict[str, list[float]] = {m: [] for m in build_map}
#     for _ in range(repeats):
#         for method in build_map:
#             t0 = time.time()
#             _block_until_ready_pytree(vjp_fns[method](bars[method]))
#             bwd_times[method].append(time.time() - t0)

#     return {k: np.array(v) for k, v in fwd_times.items()}, {
#         k: np.array(v) for k, v in bwd_times.items()
#     }


# def benchmark(matrix_sizes, matvec_nums, methods: list[str], repeats=5):
#     # Prepare per-method containers for means/stds
#     fwd_mean = {m: np.zeros((len(matrix_sizes), len(matvec_nums))) for m in methods}
#     fwd_std = {m: np.zeros((len(matrix_sizes), len(matvec_nums))) for m in methods}
#     bwd_mean = {m: np.zeros((len(matrix_sizes), len(matvec_nums))) for m in methods}
#     bwd_std = {m: np.zeros((len(matrix_sizes), len(matvec_nums))) for m in methods}

#     total = len(matrix_sizes) * len(matvec_nums)
#     pbar = tqdm(total=total, desc="Benchmarking")
#     for i, size in enumerate(matrix_sizes):
#         for j, k in enumerate(matvec_nums):
#             pbar.set_description(f"n=m={size}, k={k}")
#             fwd_times, bwd_times = _run_benchmark_for_config(
#                 int(size), int(k), repeats, methods
#             )
#             for m in methods:
#                 fwd_mean[m][i, j] = fwd_times[m].mean()
#                 fwd_std[m][i, j] = fwd_times[m].std()
#                 bwd_mean[m][i, j] = bwd_times[m].mean()
#                 bwd_std[m][i, j] = bwd_times[m].std()
#                 pbar.update(1)
#     pbar.close()
#     return {
#         "methods": methods,
#         "fwd_mean": fwd_mean,
#         "fwd_std": fwd_std,
#         "bwd_mean": bwd_mean,
#         "bwd_std": bwd_std,
#     }


# hess_aug, bidiag
if __name__ == "__main__":
    # Generate profiles for all combinations of parameters

    profiles = []
    # for alg in ["bidiag"]:  # ["hess_aug", "bidiag"]:
    #     for reorth in [True]:
    #         for custom_vjp in [True, False]:
    #             for n in [300]:
    #                 for k in np.linspace(200, 300, 5, dtype=int):
    #                     profiles.append(
    #                         {
    #                             "algorithm": alg,
    #                             "reorthogonalize": reorth,
    #                             "custom_vjp": custom_vjp,
    #                             # "m": int(n),
    #                             "n": int(n),
    #                             "k": int(k),
    #                             "matrix": "normal",
    #                             "dtype": "float64",
    #                             "seed": 0,
    #                         }
    #                     )

    # Also run the same sweeps for selected SuiteSparse matrices (shape implied by file)
    for alg in ["bidiag"]:
        for reorth in [True]:
            for custom_vjp in [True]:
                for matrix_name in ["1138_bus"]:
                    for k in np.linspace(20, 250, 4, dtype=int):
                        profiles.append(
                            {
                                "algorithm": alg,
                                "reorthogonalize": reorth,
                                "custom_vjp": custom_vjp,
                                "k": int(k),
                                "matrix": matrix_name,
                                "dtype": "float32",
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
        include_true_compile=True,
    )
