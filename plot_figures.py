"""Draw every figure and table of the paper from the saved results.

Each experiment script writes its raw measurements to results/*.json; this
script turns them into latex/figures/*.pdf and latex/table_*.tex, using the
shared style of `plot_style.py`.

Usage:
    python plot_figures.py                 # everything
    python plot_figures.py precision_sweep # a subset, by name
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import plot_style
from plot_style import METHODS, REFERENCE_LINE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "latex", "figures")
TABLES_DIR = os.path.join(HERE, "latex")

FAILURE_CEILING = 1.0
DEVICE_MEMORY = ((16e9, "16 GB"), (80e9, "80 GB (A100/H100)"))


def load(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.json")) as f:
        return json.load(f)


def select(records, **conditions):
    return [r for r in records if all(r.get(k) == v for k, v in conditions.items())]


def clip_failures(errors):
    """NaNs and errors above FAILURE_CEILING are drawn at FAILURE_CEILING."""
    errors = np.asarray(errors, dtype=float)
    return np.minimum(np.where(np.isfinite(errors), errors, np.inf), FAILURE_CEILING)


def mark_failure_ceiling(ax, x_text):
    ax.axhline(FAILURE_CEILING, **REFERENCE_LINE)
    ax.text(x_text, FAILURE_CEILING * 0.6, r"NaN or error $\geq 1$", fontsize="small", color="0.4", va="top")


def save(fig, name):
    path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def full_depth_accuracy():
    """Gradient error at full depth on random normal (float64) and Toeplitz (float32) matrices."""
    results = load("full_depth_accuracy")
    experiments = results["config"]["experiments"]
    fig, axes = plot_style.figure(2, sharey=True)
    for ax, (name, title) in zip(axes, (("normal", "Random normal, float64"), ("toeplitz", "Toeplitz, float32"))):
        for method in experiments[name]["methods"]:
            rows = select(results["records"], experiment=name, method=method)
            plot_style.plot(ax, [r["n"] for r in rows], clip_failures([r["error"] for r in rows]), method)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"Matrix size $n$ (depth $k = n$)")
    mark_failure_ceiling(axes[0], experiments["normal"]["ns"][0])
    axes[1].axhline(FAILURE_CEILING, **REFERENCE_LINE)
    axes[0].set_ylabel("Gradient error")
    plot_style.figure_legend(fig, axes[0], ncols=4)
    save(fig, "fig_full_depth_accuracy")


def reorthogonalization():
    """Forward reconstruction with and without reorthonormalization; adjoint with and without reprojection."""
    results = load("full_depth_accuracy")
    fig, (ax_fwd, ax_bwd) = plot_style.figure(2)
    for method in ("forward_noreortho", "forward"):
        rows = select(results["records"], experiment="toeplitz_reconstruction", method=method)
        plot_style.plot(ax_fwd, [r["n"] for r in rows], [r["error"] for r in rows], method)
    for method in ("bidiag_adjoint_noreproj", "bidiag_adjoint"):
        rows = select(results["records"], experiment="reprojection", method=method)
        plot_style.plot(ax_bwd, [r["n"] for r in rows], [r["error"] for r in rows], method)
    ax_fwd.set_title("Forward pass, Toeplitz, float32")
    ax_fwd.set_xlabel(r"Matrix size $n$ (depth $k = n$)")
    ax_fwd.set_ylabel("Reconstruction error")
    ax_bwd.set_title(r"Adjoint, random normal $n \times n/2$, float32")
    ax_bwd.set_xlabel(r"Matrix size $n$ (depth $k = n/2$)")
    ax_bwd.set_ylabel("Gradient error")
    for ax, loc in ((ax_fwd, "center right"), (ax_bwd, "upper left")):
        ax.set_yscale("log")
        ax.legend(loc=loc)
    save(fig, "fig_reorthogonalization")


def full_depth_cost():
    """Memory and run time against matrix size at full depth.

    The compile-time sweep in the same results file is reported in the text only:
    it is flat in the depth for all methods.
    """
    results = load("full_depth_cost")
    records = results["records"]
    fig, (ax_mem, ax_run) = plot_style.figure(2)

    adjoint = sorted(select(records, sweep="run", custom_vjp=True), key=lambda r: r["n"])
    autodiff = sorted(select(records, sweep="run", custom_vjp=False), key=lambda r: r["n"])
    ran = [r for r in autodiff if "bwd_run_s" in r]
    for method, rows in (("bidiag_adjoint", adjoint), ("bidiag_autodiff", autodiff)):
        plot_style.plot(ax_mem, [r["n"] for r in rows], [r["bytes"] / 1e9 for r in rows], method)
    limit = results["config"]["memory_limit_bytes"] / 1e9
    ax_mem.axhline(limit, **REFERENCE_LINE)
    ax_mem.text(adjoint[-1]["n"], limit / 1.6, f"{limit:.0f} GB limit", fontsize="small", color="0.4", ha="right", va="top")

    plot_style.plot(ax_run, [r["n"] for r in adjoint], [r["fwd_run_s"] for r in adjoint], "forward")
    plot_style.plot(ax_run, [r["n"] for r in adjoint], [r["bwd_run_s"] for r in adjoint], "bidiag_adjoint")
    plot_style.plot(ax_run, [r["n"] for r in ran], [r["bwd_run_s"] for r in ran], "bidiag_autodiff")

    ax_mem.set_title("Memory of value and gradient")
    ax_mem.set_ylabel("Memory (GB)")
    ax_run.set_title("Run time (CPU)")
    ax_run.set_ylabel("Time per pass (s)")
    for ax in (ax_mem, ax_run):
        ax.set_yscale("log")
        ax.set_xlabel(r"Matrix size $n$ (depth $k = n$)")
    plot_style.figure_legend(fig, ax_run, ncols=3)
    save(fig, "fig_full_depth_cost")


def precision_sweep():
    """Float32 gradient error of the log-det quadrature against kappa(A) and against depth."""
    results = load("precision_sweep")
    cfg = results["config"]
    panels = (
        ("kappa", "log10_kappa", r"Condition number $\kappa(A)$", rf"Depth $k = {cfg['kappa_sweep_depth']}$"),
        ("depth", "depth", r"Krylov depth $k$", rf"$\kappa(A) = 10^{{{cfg['depth_sweep_log10_kappa']:.0f}}}$"),
    )
    fig, axes = plot_style.figure(2, sharey=True)
    for ax, (sweep, key, xlabel, title) in zip(axes, panels):
        for method in ("bidiag_adjoint", "bidiag_autodiff", "lanczos_gram", "arnoldi_augmented"):
            rows = select(results["records"], sweep=sweep, method=method)
            xs = sorted({r[key] for r in rows})
            errors = clip_failures([[r["grad_err"] for r in rows if r[key] == x] for x in xs])
            x = 10.0 ** np.array(xs) if sweep == "kappa" else np.array(xs)
            linestyle = {"bidiag_autodiff": "--", "arnoldi_augmented": ":"}.get(method, "-")
            plot_style.plot(ax, x, np.median(errors, axis=1), method, linestyle=linestyle)
            plot_style.band(ax, x, errors.min(axis=1), errors.max(axis=1), method)
        if sweep == "kappa":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    mark_failure_ceiling(axes[0], 10.0 ** min(r["log10_kappa"] for r in select(results["records"], sweep="kappa")))
    axes[1].axhline(FAILURE_CEILING, **REFERENCE_LINE)
    axes[0].set_ylabel("Relative gradient error")
    plot_style.figure_legend(fig, axes[0], ncols=4)
    save(fig, "fig_precision_sweep")


def partial_depth_cost():
    """Memory and run time of value-and-gradient against depth for the convolution operator."""
    results = load("partial_depth_cost")
    fig, (ax_mem, ax_time) = plot_style.figure(2)
    for method in ("forward", "bidiag_adjoint", "bidiag_autodiff", "lanczos_gram", "arnoldi_augmented"):
        rows = sorted(select(results["records"], method=method), key=lambda r: r["depth"])
        plot_style.plot(ax_mem, [r["depth"] for r in rows], [r["bytes"] / 1e9 for r in rows], method)
        timed = [r for r in rows if r["seconds"] is not None]
        plot_style.plot(ax_time, [r["depth"] for r in timed], [r["seconds"] for r in timed], method)
    k_min = min(r["depth"] for r in results["records"])
    for nbytes, label in DEVICE_MEMORY:
        ax_mem.axhline(nbytes / 1e9, **REFERENCE_LINE)
        ax_mem.text(k_min, nbytes / 1e9 * 1.3, label, fontsize="small", color="0.4")
    size = f"{results['config']['size']:,}".replace(",", "{,}")
    ax_mem.set_title(rf"Memory, $n = {size}$")
    ax_mem.set_ylabel("Memory (GB)")
    ax_time.set_title("Run time (CPU)")
    ax_time.set_ylabel("Time (s)")
    for ax in (ax_mem, ax_time):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"Krylov depth $k$")
    plot_style.figure_legend(fig, ax_mem, ncols=3)
    save(fig, "fig_partial_depth_cost")


def _fmt_bytes(nbytes):
    gb = nbytes / 1e9
    return f"{gb:.1f}" if gb >= 10 else f"{gb:.2f}"


def _fmt_sci(x):
    if x is None:
        return "---"
    if not np.isfinite(x):
        return "NaN"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\cdot 10^{{{int(exponent)}}}$"


def _thousands(x):
    return f"{x:,}".replace(",", "{,}")


def table_suitesparse():
    """Memory, time, and gradient error per method on the SuiteSparse matrices."""
    results = load("suitesparse")
    methods = ("forward", "bidiag_adjoint", "bidiag_autodiff", "lanczos_gram", "arnoldi_augmented")
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Matrix & Method & Memory (GB) & Time (s) & Gradient error \\",
        r"\midrule",
    ]
    for i, info in enumerate(results["matrices"]):
        recs = {r["method"]: r for r in select(results["records"], matrix=info["matrix"])}
        header = [
            rf"\texttt{{{info['matrix'].replace('_', r'\_')}}}",
            rf"${_thousands(info['rows'])} \times {_thousands(info['cols'])}$",
            rf"{info['nnz'] / 1e6:.2f}M nonzeros",
        ]
        for j, method in enumerate(methods):
            r = recs[method]
            seconds = f"{r['seconds']:.2f}" if r["seconds"] is not None else "---"
            error = "" if method == "forward" else _fmt_sci(r["grad_err"])
            label = METHODS[method]["label"]
            label = rf"\textbf{{{label}}}" if method == "bidiag_adjoint" else label
            first = header[j] if j < len(header) else ""
            lines.append(f"{first} & {label} & {_fmt_bytes(r['bytes'])} & {seconds} & {error} \\\\")
        if i < len(results["matrices"]) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("table_suitesparse", lines)


def table_lanczos_cost():
    """Forward-pass time with and without reorthogonalization, bidiagonalization vs Lanczos on A^T A."""
    results = load("lanczos_cost")
    operators = list(dict.fromkeys(r["operator"] for r in results["records"]))
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Bidiagonalization} & \multicolumn{2}{c}{Lanczos on $A^\top A$} & \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Operator & $(N{+}M)/M$ & total (s) & orthog. (s) & total (s) & orthog. (s) & orthog. ratio \\",
        r"\midrule",
    ]
    for name in operators:
        recs = {(r["method"], r["reorthogonalize"]): r for r in select(results["records"], operator=name)}
        rows, cols = recs["bidiag", True]["rows"], recs["bidiag", True]["cols"]
        bidiag = recs["bidiag", True]["seconds"]
        lanczos = recs["lanczos_gram", True]["seconds"]
        bidiag_orth = bidiag - recs["bidiag", False]["seconds"]
        lanczos_orth = lanczos - recs["lanczos_gram", False]["seconds"]
        label = r"convolution" if name == "convolution" else rf"\texttt{{{name.replace('_', r'\_')}}}"
        lines.append(
            f"{label} & {(rows + cols) / cols:.1f} & {bidiag:.2f} & {bidiag_orth:.2f} & "
            f"{lanczos:.2f} & {lanczos_orth:.2f} & {bidiag_orth / lanczos_orth:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("table_lanczos_cost", lines)


def write_table(name, lines):
    path = os.path.join(TABLES_DIR, f"{name}.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


OUTPUTS = {
    "full_depth_accuracy": full_depth_accuracy,
    "reorthogonalization": reorthogonalization,
    "full_depth_cost": full_depth_cost,
    "precision_sweep": precision_sweep,
    "partial_depth_cost": partial_depth_cost,
    "table_suitesparse": table_suitesparse,
    "table_lanczos_cost": table_lanczos_cost,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", choices=list(OUTPUTS), help="outputs to draw (default: all)")
    names = parser.parse_args().names or list(OUTPUTS)
    plot_style.use_paper_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for name in names:
        OUTPUTS[name]()


if __name__ == "__main__":
    main()
