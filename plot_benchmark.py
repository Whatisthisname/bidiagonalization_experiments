import polars as pl
import numpy as np
import matplotlib.pyplot as plt


def _flatten_records(df: pl.DataFrame) -> pl.DataFrame:
    profile = pl.col("profile").struct
    metrics = pl.col("metrics").struct
    return df.select(
        profile.field("algorithm").alias("algorithm"),
        profile.field("reorthogonalize").alias("reorthogonalize"),
        profile.field("custom_vjp").alias("custom_vjp"),
        profile.field("matrix").alias("matrix"),
        profile.field("dtype").alias("dtype"),
        profile.field("n").alias("n"),
        profile.field("k").alias("k"),
        metrics.field("fwd_steady_mean_s").alias("fwd_steady_mean_s"),
        metrics.field("fwd_steady_std_s").alias("fwd_steady_std_s"),
        metrics.field("fwd_steady_repeats").alias("fwd_steady_repeats"),
        metrics.field("bwd_steady_mean_s").alias("bwd_steady_mean_s"),
        metrics.field("bwd_steady_std_s").alias("bwd_steady_std_s"),
        metrics.field("bwd_steady_repeats").alias("bwd_steady_repeats"),
        metrics.field("fwd_compile_s").alias("fwd_compile_s"),
        metrics.field("bwd_compile_s").alias("bwd_compile_s"),
    )


def _aggregate_for_algorithm(
    df_flat: pl.DataFrame,
    algorithm: str,
    n: int,
    matrix: str = "normal",
    reorthogonalize: bool | None = None,
) -> dict:
    filt = (
        (pl.col("algorithm") == algorithm)
        & (pl.col("n") == n)
        & (pl.col("matrix") == matrix)
    )
    if reorthogonalize is not None:
        filt = filt & (pl.col("reorthogonalize") == reorthogonalize)

    df = df_flat.filter(filt)
    if df.height == 0:
        return {}

    # Since each profile configuration produces exactly one record, just extract values directly
    # Split by custom_vjp for backward times
    df_custom = (
        df.filter(pl.col("custom_vjp"))
        .select(
            [
                "k",
                "fwd_steady_mean_s",
                "fwd_steady_std_s",
                "fwd_compile_s",
                "bwd_steady_mean_s",
                "bwd_steady_std_s",
                "bwd_compile_s",
            ]
        )
        .rename(
            {
                "bwd_steady_mean_s": "bwd_custom_mean",
                "bwd_steady_std_s": "bwd_custom_std",
                "bwd_compile_s": "bwd_custom_compile",
            }
        )
        .sort("k")
    )

    df_auto = (
        df.filter(~pl.col("custom_vjp"))
        .select(["k", "bwd_steady_mean_s", "bwd_steady_std_s", "bwd_compile_s"])
        .rename(
            {
                "bwd_steady_mean_s": "bwd_auto_mean",
                "bwd_steady_std_s": "bwd_auto_std",
                "bwd_compile_s": "bwd_auto_compile",
            }
        )
        .sort("k")
    )

    # Forward times are the same regardless of custom_vjp, so take from custom_vjp=True data
    fwd_data = df_custom.select(
        ["k", "fwd_steady_mean_s", "fwd_steady_std_s", "fwd_compile_s"]
    ).rename(
        {
            "fwd_steady_mean_s": "fwd_mean",
            "fwd_steady_std_s": "fwd_std",
            "fwd_compile_s": "fwd_compile",
        }
    )

    # Join on k to align series
    out = fwd_data.join(
        df_custom.select(
            ["k", "bwd_custom_mean", "bwd_custom_std", "bwd_custom_compile"]
        ),
        on="k",
        how="left",
    ).join(
        df_auto.select(["k", "bwd_auto_mean", "bwd_auto_std", "bwd_auto_compile"]),
        on="k",
        how="left",
    )
    return {"algorithm": algorithm, "data": out}


def _aggregate_for_algorithm_by_matrix(
    df_flat: pl.DataFrame,
    algorithm: str,
    matrix: str,
    reorthogonalize: bool | None = None,
) -> dict:
    filt = (pl.col("algorithm") == algorithm) & (pl.col("matrix") == matrix)
    if reorthogonalize is not None:
        filt = filt & (pl.col("reorthogonalize") == reorthogonalize)

    df = df_flat.filter(filt)
    if df.height == 0:
        return {}

    # Since each profile configuration produces exactly one record, just extract values directly
    # Split by custom_vjp for backward times
    df_custom = (
        df.filter(pl.col("custom_vjp"))
        .select(
            [
                "k",
                "fwd_steady_mean_s",
                "fwd_steady_std_s",
                "fwd_compile_s",
                "bwd_steady_mean_s",
                "bwd_steady_std_s",
                "bwd_compile_s",
            ]
        )
        .rename(
            {
                "bwd_steady_mean_s": "bwd_custom_mean",
                "bwd_steady_std_s": "bwd_custom_std",
                "bwd_compile_s": "bwd_custom_compile",
            }
        )
        .sort("k")
    )

    df_auto = (
        df.filter(~pl.col("custom_vjp"))
        .select(["k", "bwd_steady_mean_s", "bwd_steady_std_s", "bwd_compile_s"])
        .rename(
            {
                "bwd_steady_mean_s": "bwd_auto_mean",
                "bwd_steady_std_s": "bwd_auto_std",
                "bwd_compile_s": "bwd_auto_compile",
            }
        )
        .sort("k")
    )

    # Forward times are the same regardless of custom_vjp, so take from custom_vjp=True data
    fwd_data = df_custom.select(
        ["k", "fwd_steady_mean_s", "fwd_steady_std_s", "fwd_compile_s"]
    ).rename(
        {
            "fwd_steady_mean_s": "fwd_mean",
            "fwd_steady_std_s": "fwd_std",
            "fwd_compile_s": "fwd_compile",
        }
    )

    # Join on k to align series
    out = fwd_data.join(
        df_custom.select(
            ["k", "bwd_custom_mean", "bwd_custom_std", "bwd_custom_compile"]
        ),
        on="k",
        how="left",
    ).join(
        df_auto.select(["k", "bwd_auto_mean", "bwd_auto_std", "bwd_auto_compile"]),
        on="k",
        how="left",
    )
    return {"algorithm": algorithm, "data": out}


def plot_profiles(
    jsonl_path: str,
    n: int,
    matrix_type: str = "normal",
    reorthogonalize: bool | None = None,
):
    df = pl.read_ndjson(jsonl_path)
    df_flat = _flatten_records(df)

    algs = ["hess_aug", "bidiag"]
    for alg in algs:
        # Maintain backward compatibility: parameter is named matrix_type, but we filter on normalized 'matrix'
        agg = _aggregate_for_algorithm(
            df_flat, alg, n=n, matrix=matrix_type, reorthogonalize=reorthogonalize
        )
        if not agg:
            print(
                f"No data for algorithm={alg}, n={n}, matrix_type={matrix_type}, reorthogonalize={reorthogonalize}"
            )
            continue
        data = agg["data"]

        k = data["k"].to_numpy()
        forward = data["fwd_mean"].to_numpy()
        forward_std = np.nan_to_num(data.get_column("fwd_std").to_numpy(), nan=0.0)
        adjoint = data["bwd_custom_mean"].to_numpy()
        adjoint_std = np.nan_to_num(
            data.get_column("bwd_custom_std").to_numpy(), nan=0.0
        )
        backprop = data["bwd_auto_mean"].to_numpy()
        backprop_std = np.nan_to_num(
            data.get_column("bwd_auto_std").to_numpy(), nan=0.0
        )

        forward_c = data["fwd_compile"].to_numpy()
        adjoint_c = data["bwd_custom_compile"].to_numpy()
        backprop_c = data["bwd_auto_compile"].to_numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        title_prefix = "Hessenberg (aug)" if alg == "hess_aug" else "Bidiagonalization"

        # Runtime subplot
        ax1.set_title(f"{title_prefix}: Run time")
        ax1.errorbar(
            k, forward, yerr=forward_std, linestyle="--", color="black", label="Forward"
        )
        ax1.errorbar(
            k, adjoint, yerr=adjoint_std, color="#1f77b4", label="Adjoint (custom)"
        )
        ax1.errorbar(
            k, backprop, yerr=backprop_std, color="#ff7f0e", label="Backprop (autodiff)"
        )
        ax1.set_xlabel("Krylov-space depth k")
        ax1.set_ylabel("Wall time (sec)")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Compile subplot
        ax2.set_title(f"{title_prefix}: Compile time")
        ax2.plot(k, forward_c, linestyle="--", color="black", label="Forward")
        # ax2.set_yscale("log")
        ax2.plot(k, adjoint_c, color="#1f77b4", label="Adjoint (custom)")
        ax2.plot(k, backprop_c, color="#ff7f0e", label="Backprop (autodiff)")
        ax2.set_xlabel("Krylov-space depth k")
        ax2.set_ylabel("Wall time (sec)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()


def plot_profiles_for_sparse_matrix(
    jsonl_path: str,
    matrix: str,
    reorthogonalize: bool | None = None,
):
    df = pl.read_ndjson(jsonl_path)
    df_flat = _flatten_records(df)

    algs = ["hess_aug", "bidiag"]
    for alg in algs:
        agg = _aggregate_for_algorithm_by_matrix(
            df_flat, alg, matrix=matrix, reorthogonalize=reorthogonalize
        )
        if not agg:
            print(
                f"No data for algorithm={alg}, matrix={matrix}, reorthogonalize={reorthogonalize}"
            )
            continue
        data = agg["data"]

        k = data["k"].to_numpy()
        forward = data["fwd_mean"].to_numpy()
        forward_std = np.nan_to_num(data.get_column("fwd_std").to_numpy(), nan=0.0)
        adjoint = data["bwd_custom_mean"].to_numpy()
        adjoint_std = np.nan_to_num(
            data.get_column("bwd_custom_std").to_numpy(), nan=0.0
        )
        backprop = data["bwd_auto_mean"].to_numpy()
        backprop_std = np.nan_to_num(
            data.get_column("bwd_auto_std").to_numpy(), nan=0.0
        )

        forward_c = data["fwd_compile"].to_numpy()
        adjoint_c = data["bwd_custom_compile"].to_numpy()
        backprop_c = data["bwd_auto_compile"].to_numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        title_prefix = "Hessenberg (aug)" if alg == "hess_aug" else "Bidiagonalization"

        ax1.set_title(f"{title_prefix}: Run time on {matrix}")
        ax1.errorbar(
            k, forward, yerr=forward_std, linestyle="--", color="black", label="Forward"
        )
        ax1.errorbar(
            k, adjoint, yerr=adjoint_std, color="#1f77b4", label="Adjoint (custom)"
        )
        ax1.errorbar(
            k, backprop, yerr=backprop_std, color="#ff7f0e", label="Backprop (autodiff)"
        )
        ax1.set_xlabel("Krylov-space depth k")
        ax1.set_ylabel("Wall time (sec)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title(f"{title_prefix}: Compile time on {matrix}")
        ax2.plot(k, forward_c, linestyle="--", color="black", label="Forward")
        ax2.plot(k, adjoint_c, color="#1f77b4", label="Adjoint (custom)")
        ax2.plot(k, backprop_c, color="#ff7f0e", label="Backprop (autodiff)")
        ax2.set_xlabel("Krylov-space depth k")
        ax2.set_ylabel("Wall time (sec)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage: adjust n and filters as needed
    plot_profiles(
        jsonl_path="benchmarks.ndjson",
        n=1138,
        matrix_type="1138_bus",  # 1138_bus, normal
        reorthogonalize=True,
    )
