import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt


def _flatten_records(df: pl.DataFrame) -> pl.DataFrame:
    profile = pl.col("profile").struct
    metrics = pl.col("metrics").struct
    return df.select(
        profile.field("algorithm"),
        profile.field("reorthogonalize"),
        profile.field("custom_vjp"),
        profile.field("matrix"),
        profile.field("dtype"),
        profile.field("n"),
        profile.field("k"),
        metrics.field("fwd_steady_mean_s"),
        metrics.field("fwd_steady_std_s"),
        metrics.field("fwd_steady_repeats"),
        metrics.field("bwd_steady_mean_s"),
        metrics.field("bwd_steady_std_s"),
        metrics.field("bwd_steady_repeats"),
        metrics.field("fwd_compile_s"),
        metrics.field("bwd_compile_s"),
    )


def _aggregate_by_k(
    df_flat: pl.DataFrame,
    algorithm: str,
    number: int,
    number_name: str = "n",
    group_by_name: str = "k",
    matrix: str = "normal",
    reorthogonalize: bool | None = None,
) -> dict:
    """Take the dataframe, filter on 'number_name' = 'number', and then group by 'group_by_names'"""

    filt = (
        (pl.col("algorithm") == algorithm)
        & (pl.col(number_name) == number)
        & (pl.col("matrix") == matrix)
    )
    if reorthogonalize is not None:
        filt = filt & (pl.col("reorthogonalize") == reorthogonalize)

    df = df_flat.filter(filt)
    if df.height == 0:
        return {}

    # Since each profile configuration produces exactly one record, just extract values directly
    # Split by custom_vjp for backward times
    df_custom_vjp = (
        df.filter(pl.col("custom_vjp"))
        .select(
            [
                group_by_name,
                "fwd_steady_mean_s",
                "fwd_steady_std_s",
                "fwd_compile_s",
                "bwd_steady_mean_s",
                "bwd_steady_std_s",
                "bwd_compile_s",
            ]
        )
        .sort(group_by_name)
    )

    df_autodiff = (
        df.filter(~pl.col("custom_vjp"))
        .select(
            [group_by_name, "bwd_steady_mean_s", "bwd_steady_std_s", "bwd_compile_s"]
        )
        .sort(group_by_name)
    )

    # Forward times are the same regardless of custom_vjp, so take from custom_vjp=True data
    fwd_data = df_custom_vjp.select(
        [group_by_name, "fwd_steady_mean_s", "fwd_steady_std_s", "fwd_compile_s"]
    )

    # Join on k to align series
    out = fwd_data.join(
        df_custom_vjp.select(
            [
                group_by_name,
                pl.col("bwd_steady_mean_s").alias("bwd_steady_mean_s_customvjp"),
                pl.col("bwd_steady_std_s").alias("bwd_steady_std_s_customvjp"),
                pl.col("bwd_compile_s").alias("bwd_compile_s_customvjp"),
            ]
        ),
        on=group_by_name,
        how="left",
    ).join(
        df_autodiff.select(
            [
                group_by_name,
                pl.col("bwd_steady_mean_s").alias("bwd_steady_mean_s_auto"),
                pl.col("bwd_steady_std_s").alias("bwd_steady_std_s_auto"),
                pl.col("bwd_compile_s").alias("bwd_compile_s_auto"),
            ]
        ),
        on=group_by_name,
        how="left",
    )
    return {"algorithm": algorithm, "data": out}


def plot_profiles(
    jsonl_path: str,
    n: int | None,
    k: int | None,
    matrix_type: str = "normal",
    reorthogonalize: bool | None = None,
):
    # only one may be defined of 'n' or 'k'
    assert (n is not None and k is None) or (k is not None and n is None)

    if n is not None:
        number = n
        number_name = "n"
        group_by_name = "k"
        xlabel = "Krylov-space depth k"
    else:
        number = k
        number_name = "k"
        group_by_name = "n"
        xlabel = "Matrix height n"

    df = pl.read_ndjson(jsonl_path)
    df_flat = _flatten_records(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    algs = ["bidiag", "hess_aug"]
    for alg in algs:
        method_name = "Aug. Hess." if alg == "hess_aug" else "Bidiag."
        style = "dashed" if alg == "hess_aug" else "solid"

        agg = _aggregate_by_k(
            df_flat,
            alg,
            number=number,
            number_name=number_name,
            group_by_name=group_by_name,
            matrix=matrix_type,
            reorthogonalize=reorthogonalize,
        )
        if not agg:
            print(
                f"No data for algorithm={alg}, number={number}, matrix_type={matrix_type}, reorthogonalize={reorthogonalize}"
            )
            continue
        data = agg["data"]

        group_by_value = data[group_by_name].to_numpy()
        forward_run_mean = data["fwd_steady_mean_s"].to_numpy()
        forward_run_std = np.nan_to_num(
            data.get_column("fwd_steady_std_s").to_numpy(), nan=0.0
        )
        adjoint_run_mean = data["bwd_steady_mean_s_customvjp"].to_numpy()
        adjoint_run_std = np.nan_to_num(
            data.get_column("bwd_steady_std_s_customvjp").to_numpy(), nan=0.0
        )

        backprop_run_mean = data["bwd_steady_mean_s_auto"].to_numpy()
        backprop_run_std = np.nan_to_num(
            data.get_column("bwd_steady_std_s_auto").to_numpy(), nan=0.0
        )

        forward_compile = data["fwd_compile_s"].to_numpy()
        adjoint_compile = data["bwd_compile_s_customvjp"].to_numpy()
        backprop_compile = data["bwd_compile_s_auto"].to_numpy()

        # Runtime subplot
        ax1.set_title("Run time")
        ax1.errorbar(
            group_by_value,
            forward_run_mean,
            yerr=forward_run_std,
            linestyle=style,
            color="black",
            label=f"{method_name} Forward",
        )
        ax1.errorbar(
            group_by_value,
            adjoint_run_mean,
            yerr=adjoint_run_std,
            linestyle=style,
            color="#1f77b4",
            label=f"{method_name} Adjoint",
        )
        ax1.errorbar(
            group_by_value,
            backprop_run_mean,
            yerr=backprop_run_std,
            linestyle=style,
            color="#ff7f0e",
            label=f"{method_name} Autodiff",
        )

        # Compile subplot
        ax2.set_title("Compile time")
        ax2.plot(
            group_by_value,
            forward_compile,
            linestyle=style,
            color="black",
            label=f"{method_name} Forward",
        )
        ax2.plot(
            group_by_value,
            adjoint_compile,
            linestyle=style,
            color="#1f77b4",
            label=f"{method_name} Adjoint",
        )
        ax2.plot(
            group_by_value,
            backprop_compile,
            linestyle=style,
            color="#ff7f0e",
            label=f"{method_name} Autodiff",
        )

        for ax in (ax1, ax2):
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Wall time (sec)")
            ax.set_yscale("log")
            ### adding more dense yticklabels
            ticks = ax.get_yticks()
            labels = []
            for t in ticks:
                if t == 0 or np.isnan(t):
                    labels.append("0")
                else:
                    labels.append(f"{t:.2g}")
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ###
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.show()


FORWARD_COLOR = "black"
ADJOINT_COLOR = "#1f77b4"
AUTODIFF_COLOR = "#ff7f0e"


def _figures_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "latex", "figures")
    os.makedirs(out, exist_ok=True)
    return out


def _bidiag_slice(df_flat, matrix, reorthogonalize):
    return df_flat.filter(
        (pl.col("algorithm") == "bidiag")
        & (pl.col("matrix") == matrix)
        & (pl.col("reorthogonalize") == reorthogonalize)
    )


def make_compile_time_figure(jsonl_path, n=950, matrix="diagonal", reorthogonalize=True):
    """Compile time vs Krylov depth k at a fixed matrix size."""
    df_flat = _flatten_records(pl.read_ndjson(jsonl_path))
    sub = _bidiag_slice(df_flat, matrix, reorthogonalize).filter(pl.col("n") == n)

    cvjp = sub.filter(pl.col("custom_vjp")).sort("k")
    auto = sub.filter(~pl.col("custom_vjp")).sort("k")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(cvjp["k"], cvjp["fwd_compile_s"], "-o", color=FORWARD_COLOR, label="Forward")
    ax.plot(cvjp["k"], cvjp["bwd_compile_s"], "-o", color=ADJOINT_COLOR, label="Adjoint")
    ax.plot(auto["k"], auto["bwd_compile_s"], "-o", color=AUTODIFF_COLOR, label="Autodiff")
    ax.set_xlabel("Krylov-space depth $k$")
    ax.set_ylabel("Compile time (s)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_backward_time_figure(jsonl_path, matrix="normal", reorthogonalize=True):
    """Backward-pass run time vs matrix size along the k=n diagonal.

    The autodiff series stops where it ran out of memory (missing records),
    while the adjoint continues to larger problems.
    """
    df_flat = _flatten_records(pl.read_ndjson(jsonl_path))
    sub = _bidiag_slice(df_flat, matrix, reorthogonalize).filter(pl.col("k") == pl.col("n"))

    cvjp = sub.filter(pl.col("custom_vjp")).sort("n")
    auto = (
        sub.filter(~pl.col("custom_vjp"))
        .filter(pl.col("bwd_steady_mean_s").is_not_null())
        .sort("n")
    )

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(cvjp["n"], cvjp["fwd_steady_mean_s"], "-o", color=FORWARD_COLOR, label="Forward")
    ax.plot(cvjp["n"], cvjp["bwd_steady_mean_s"], "-o", color=ADJOINT_COLOR, label="Adjoint")
    ax.plot(auto["n"], auto["bwd_steady_mean_s"], "-o", color=AUTODIFF_COLOR, label="Autodiff")
    if auto.height:
        last = auto.sort("n").tail(1)
        ax.annotate(
            "out of memory",
            xy=(last["n"][0], last["bwd_steady_mean_s"][0]),
            xytext=(8, -2),
            textcoords="offset points",
            color=AUTODIFF_COLOR,
            fontsize="small",
        )
    ax.set_xlabel("Matrix size $n$ (with $k=n$)")
    ax.set_ylabel("Backward-pass time (s)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    out = _figures_dir()
    fig_compile = make_compile_time_figure("benchmarks.ndjson")
    fig_compile.savefig(os.path.join(out, "fig_compile_time.pdf"), bbox_inches="tight")
    fig_backward = make_backward_time_figure("benchmarks.ndjson")
    fig_backward.savefig(os.path.join(out, "fig_backward_time.pdf"), bbox_inches="tight")
    print("SAVED_PERF_FIGURES")
