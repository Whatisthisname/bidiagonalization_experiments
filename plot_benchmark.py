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
        profile.field("matrix_type").alias("matrix_type"),
        profile.field("dtype").alias("dtype"),
        profile.field("n").alias("n"),
        profile.field("k").alias("k"),
        metrics.field("fwd_first_s").alias("fwd_first_s"),
        metrics.field("fwd_second_s").alias("fwd_second_s"),
        metrics.field("fwd_steady_mean_s").alias("fwd_steady_mean_s"),
        metrics.field("fwd_steady_std_s").alias("fwd_steady_std_s"),
        metrics.field("fwd_steady_repeats").alias("fwd_steady_repeats"),
        metrics.field("bwd_first_s").alias("bwd_first_s"),
        metrics.field("bwd_second_s").alias("bwd_second_s"),
        metrics.field("bwd_steady_mean_s").alias("bwd_steady_mean_s"),
        metrics.field("bwd_steady_std_s").alias("bwd_steady_std_s"),
        metrics.field("bwd_steady_repeats").alias("bwd_steady_repeats"),
        metrics.field("fwd_compile_s").alias("fwd_compile_s"),
        metrics.field("bwd_compile_s").alias("bwd_compile_s"),
    ).with_columns(
        # Fallback compile time estimates if explicit compile times are missing
        pl.when(pl.col("fwd_compile_s").is_null())
        .then(pl.col("fwd_first_s") - pl.col("fwd_second_s"))
        .otherwise(pl.col("fwd_compile_s"))
        .alias("fwd_compile_eff_s"),
        pl.when(pl.col("bwd_compile_s").is_null())
        .then(pl.col("bwd_first_s") - pl.col("bwd_second_s"))
        .otherwise(pl.col("bwd_compile_s"))
        .alias("bwd_compile_eff_s"),
    )


def _aggregate_for_algorithm(
    df_flat: pl.DataFrame,
    algorithm: str,
    n: int,
    matrix_type: str = "normal",
    reorthogonalize: bool | None = None,
) -> dict:
    filt = (
        (pl.col("algorithm") == algorithm)
        & (pl.col("n") == n)
        & (pl.col("matrix_type") == matrix_type)
    )
    if reorthogonalize is not None:
        filt = filt & (pl.col("reorthogonalize") == reorthogonalize)

    df = df_flat.filter(filt)
    if df.height == 0:
        return {}

    # Forward times (independent of custom_vjp) — average across both flags if present
    fwd = (
        df.with_columns(
            (pl.col("fwd_steady_std_s") ** 2).alias("fwd_var"),
            # default repeats=5 if missing, so dof=4
            (pl.coalesce([pl.col("fwd_steady_repeats"), pl.lit(5)]) - 1).alias(
                "fwd_dof"
            ),
        )
        .group_by("k")
        .agg(
            pl.mean("fwd_steady_mean_s").alias("fwd_mean"),
            # pooled within-run std: sqrt( sum((n_i-1)*s_i^2) / sum(n_i-1) )
            (
                (pl.col("fwd_dof") * pl.col("fwd_var")).sum() / pl.col("fwd_dof").sum()
            ).alias("fwd_pooled_var"),
            pl.mean("fwd_compile_eff_s").alias("fwd_compile"),
        )
        .with_columns(pl.col("fwd_pooled_var").sqrt().alias("fwd_std"))
        .sort("k")
    )

    # Backward times split by custom_vjp
    bwd_custom = (
        df.filter(pl.col("custom_vjp"))
        .with_columns(
            (pl.col("bwd_steady_std_s") ** 2).alias("bwd_var"),
            (pl.coalesce([pl.col("bwd_steady_repeats"), pl.lit(5)]) - 1).alias(
                "bwd_dof"
            ),
        )
        .group_by("k")
        .agg(
            pl.mean("bwd_steady_mean_s").alias("bwd_custom_mean"),
            (
                (pl.col("bwd_dof") * pl.col("bwd_var")).sum() / pl.col("bwd_dof").sum()
            ).alias("bwd_custom_pooled_var"),
            pl.mean("bwd_compile_eff_s").alias("bwd_custom_compile"),
        )
        .with_columns(pl.col("bwd_custom_pooled_var").sqrt().alias("bwd_custom_std"))
        .sort("k")
    )
    bwd_auto = (
        df.filter(~pl.col("custom_vjp"))
        .with_columns(
            (pl.col("bwd_steady_std_s") ** 2).alias("bwd_var"),
            (pl.coalesce([pl.col("bwd_steady_repeats"), pl.lit(5)]) - 1).alias(
                "bwd_dof"
            ),
        )
        .group_by("k")
        .agg(
            pl.mean("bwd_steady_mean_s").alias("bwd_auto_mean"),
            (
                (pl.col("bwd_dof") * pl.col("bwd_var")).sum() / pl.col("bwd_dof").sum()
            ).alias("bwd_auto_pooled_var"),
            pl.mean("bwd_compile_eff_s").alias("bwd_auto_compile"),
        )
        .with_columns(pl.col("bwd_auto_pooled_var").sqrt().alias("bwd_auto_std"))
        .sort("k")
    )

    # Join on k to align series
    out = fwd.join(bwd_custom, on="k", how="left").join(bwd_auto, on="k", how="left")
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
        agg = _aggregate_for_algorithm(
            df_flat, alg, n=n, matrix_type=matrix_type, reorthogonalize=reorthogonalize
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
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Compile subplot
        ax2.set_title(f"{title_prefix}: Compile time")
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
        n=500,
        matrix_type="normal",
        reorthogonalize=True,
    )
