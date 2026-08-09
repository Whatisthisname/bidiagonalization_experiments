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

To regenerate benchmark data before plotting compile/backward-time figures:

```bash
python experiment_benchmark.py
python experiment_plot_benchmark.py
```

The accuracy figures are produced by `make_stability_figure()`, `make_hilbert_stability_figure()`, `make_hilbert_reconstruction_figure()`, and `make_reprojection_figure()` in `experiment_check_reconstruction_properties.py`. Running the script regenerates all four accuracy PDFs in one pass.
