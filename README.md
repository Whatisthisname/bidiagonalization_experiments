# bidiagonalization_experiments

Numerically stable gradients of Golub-Kahan bidiagonalization.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the figures and tables

Every experiment script only computes and writes its raw measurements to `results/*.json`. A single script draws all figures (`latex/figures/*.pdf`) and tables (`latex/table_*.tex`) from these files:

```bash
python plot_figures.py                       # everything
python plot_figures.py precision_sweep       # one output, by name
```

The shared style lives in `plot_style.py`: tueplots font sizes and line widths, Computer Modern via LaTeX, and one color, marker, and label per method. Figures are drawn at the size at which they appear in `latex/document.tex` (whose `\textwidth` is 460.72 pt) and are included there without scaling.

| Output | Experiment script | Raw data | Notes |
|--------|-------------------|----------|-------|
| `fig_full_depth_accuracy.pdf` | `experiment_full_depth_accuracy.py` | `results/full_depth_accuracy.json` | Gradient error of the full-depth reconstruction on random normal (float64) and Cauchy (float32) matrices (~2 min) |
| `fig_reorthogonalization.pdf` | `experiment_full_depth_accuracy.py` | `results/full_depth_accuracy.json` | Forward reconstruction with and without reorthonormalization; adjoint with and without reprojection |
| `fig_full_depth_cost.pdf` | `experiment_full_depth_cost.py` | `results/full_depth_cost.json` | Compile time vs depth, run time vs size at full depth (~10 min). Autodiff is skipped above 8 GB |
| `fig_precision_sweep.pdf` | `experiment_precision_sweep.py` | `results/precision_sweep.json` | Float32 gradient error of the log-det quadrature vs $\kappa(A)$ and vs depth; four methods, $1000 \times 500$, 5 seeds (~4 min) |
| `fig_partial_depth_cost.pdf` | `experiment_partial_depth_cost.py` | `results/partial_depth_cost.json` | Memory (XLA buffer assignment) and CPU time vs depth for a convolution operator, $n = 20{,}000$ (~4 min). Programs above 12 GB are not timed |
| `table_suitesparse.tex` | `experiment_suitesparse.py` | `results/suitesparse.json` | Memory, time, and float32 gradient error at $k = 100$ on five SuiteSparse matrices; downloads them into `data/suitesparse` (~470 MB), ~15 min. Programs above 8 GB are not run; `--fresh` recomputes all matrices |
| `table_lanczos_cost.tex` | `experiment_lanczos_cost.py` | `results/lanczos_cost.json` | Forward-pass time of bidiagonalization and Lanczos on $A^\top A$ with and without reorthogonalization (~3 min; needs the SuiteSparse matrices) |

`experiment_benchmark.py` and `benchmarks.ndjson` are an older, broader benchmark (including the augmented Hessenberg formulation). Its data predates the fix of the reorthogonalization projector in `algo_bidiag.py`, and the paper no longer uses it.
