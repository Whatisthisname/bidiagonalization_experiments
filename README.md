# bidiagonalization_experiments

Numerically stable gradients of Golub-Kahan bidiagonalization.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the figures

All paper figures live in `latex/figures/`.

| Figure | Script | Notes |
|--------|--------|-------|
| `fig_stability.pdf` | `python experiment_check_reconstruction_properties.py` | Gradient error vs matrix size (random $n \times n$ matrices) |
| `fig_hilbert_stability.pdf` | `python experiment_check_reconstruction_properties.py` | Gradient error on Cauchy matrices ($C_{ij}=1/(1+|i-j|)$, $k=n$, float32) |
| `fig_hilbert_reconstruction.pdf` | `python experiment_check_reconstruction_properties.py` | Forward reconstruction error on Cauchy matrices ($k=n$, float32) |
| `fig_reprojection.pdf` | `python experiment_check_reconstruction_properties.py` | Adjoint accuracy with vs without reprojection ($n \times (n/2)$ random matrices, $k=n/2$, float32) |
| `fig_compile_time.pdf` | `python experiment_plot_benchmark.py` | Requires `benchmarks.ndjson` |
| `fig_backward_time.pdf` | `python experiment_plot_benchmark.py` | Requires `benchmarks.ndjson` |
| `fig_precision_sweep.pdf` | `python experiment_precision_sweep.py` | Float32 gradient error of the log-det quadrature $v^\top \log(A^\top A) v$ vs $\kappa(A)$ and vs depth $k$; bidiag adjoint, bidiag autodiff, Lanczos on $A^\top A$, augmented Arnoldi ($1000 \times 500$, 5 seeds, ~4 min). Raw data in `results_precision_sweep.json`; `--plot-only` re-plots |
| `fig_partial_depth_cost.pdf` | `python experiment_partial_depth_cost.py` | Memory (XLA buffer assignment) and CPU time vs depth $k$ for a matrix-free convolution operator, $n = 20{,}000$ (~4 min). Programs above 12 GB are not timed. Raw data in `results_partial_depth_cost.json`; `--plot-only` re-plots |

To regenerate benchmark data before plotting compile/backward-time figures:

```bash
python experiment_benchmark.py
python experiment_plot_benchmark.py
```

The accuracy figures are produced by `make_stability_figure()`, `make_hilbert_stability_figure()`, `make_hilbert_reconstruction_figure()`, and `make_reprojection_figure()` in `experiment_check_reconstruction_properties.py`. Running the script regenerates all four accuracy PDFs in one pass.
