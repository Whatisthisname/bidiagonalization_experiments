import dataclasses
import os
from collections.abc import Callable
from typing import Literal

import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from tueplots import axes, figsizes, fontsizes

from algo_bidiag import bidiagonalize as bidiagonalize
from algo_hessenberg import hessenberg


def hilbert_matrix(ndim, /):
    a = jnp.arange(ndim)
    return 1 / (1 + (a[:, None] + a[None, :]))


def cauchy_matrix(ndim, /):
    """Symmetric Cauchy matrix C_ij = 1/(1 + |i-j|).

    Same smooth, positive-definite, ill-conditioned flavour as the Hilbert
    matrix but with condition number O(n) instead of exponential in n.
    """
    idx = jnp.arange(ndim)
    return 1 / (1 + jnp.abs(idx[:, None] - idx[None, :]))


def normal_matrix(ndim):
    seed = 1
    return jax.random.normal(key=jax.random.PRNGKey(seed), shape=(ndim, ndim))


plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())
plt.rcParams.update(figsizes.iclr2024(rel_width=0.4, height_to_width_ratio=1.25))
plt.rcParams.update(fontsizes.iclr2024(default_smaller=2))

key = jax.random.PRNGKey(1)
key_A, key_v = jax.random.split(key, num=2)


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    algo: Literal["Hess", "Bidi"]
    reortho: bool
    adjoint: bool
    reproj_adj: Literal["match", "none"] | None = None
    reproj_repeats: int = 1


hess_backprop = Config(algo="Hess", adjoint=False, reortho=False)
bidi_backprop = Config(algo="Bidi", adjoint=False, reortho=False)
hess_backprop_reo = Config(algo="Hess", adjoint=False, reortho=True)
bidi_backprop_reo = Config(algo="Bidi", adjoint=False, reortho=True)
hess_adjoint_reo = Config(algo="Hess", adjoint=True, reortho=True, reproj_adj="none")
bidi_adjoint_reo = Config(algo="Bidi", adjoint=True, reortho=True, reproj_adj="none")
hess_adjoint_rep = Config(algo="Hess", adjoint=True, reortho=True, reproj_adj="match")
bidi_adjoint_rep = Config(algo="Bidi", adjoint=True, reortho=True, reproj_adj="match")
bidi_adjoint_rep2 = Config(algo="Bidi", adjoint=True, reortho=True, reproj_adj="match", reproj_repeats=2)


def run_experiment(
    configuration,
    *,
    ns: jnp.ndarray,
    matrix_fn: Callable[[int], jnp.ndarray],
    xlabel: str,
    ylabel: str,
    style_overrides: dict[Config, dict] | None = None,
    width_fn: Callable[[int], int] | None = None,
    num_matvecs_fn: Callable[[int, int], int] | None = None,
    reproj_repeats_fn: Callable[[Config], int] | None = None,
    dtype=jnp.float64,
):
    if width_fn is None:
        width_fn = lambda n: n
    if num_matvecs_fn is None:
        num_matvecs_fn = lambda _n, width: width
    if reproj_repeats_fn is None:
        reproj_repeats_fn = lambda config: config.reproj_repeats

    fig, ax = plt.subplots(figsize=(5, 3.2))

    for config, label in tqdm.tqdm(configuration[1].items(), desc="Testing setups"):
        reconstruct_loss = []
        jacobian_loss = []

        for n in tqdm.tqdm(ns, desc=f"Testing {label}", leave=False):
            height = int(n)
            width = int(width_fn(height))
            num_matvecs = int(num_matvecs_fn(height, width))
            A = matrix_fn(height)[:, :width].astype(dtype)
            v = jax.random.normal(key_v, shape=(width,)).astype(dtype)
            flat, unflatten = jax.flatten_util.ravel_pytree(A)

            if config.algo == "Bidi":
                bd_func = bidiagonalize(
                    num_matvecs=num_matvecs,
                    custom_vjp=config.adjoint,
                    reorthogonalize=config.reortho,
                    also_reorthogonalize_vjp=(config.reproj_adj == "match"),
                    reproj_repeats=reproj_repeats_fn(config),
                )

                @jax.jit
                def decompose_reconstruct(x):
                    a = unflatten(x)

                    result = bd_func(lambda v, p: p @ v, v, a)
                    ls, rs, alphas, betas, res = (
                        result.ls,
                        result.rs,
                        result.as_,
                        result.bs,
                        result.res,
                    )

                    B = jnp.diag(alphas) + jnp.diag(betas, k=1)
                    return jax.flatten_util.ravel_pytree(ls @ B @ rs.T)[0]

            else:  # if config.algo == "Hess":

                def matvec_sym(v0, *params):
                    (A,) = params
                    upper, lower = jnp.split(v0, [height])
                    return jnp.concat((A @ lower, A.T @ upper))

                v_aug = jnp.concat([jnp.zeros(height, dtype=dtype), v])

                forward_reortho = "full" if config.reortho else "none"
                adjoint_reortho = config.reproj_adj if config.reproj_adj else "none"
                hess_func = hessenberg(
                    height + width,
                    reortho=adjoint_reortho,
                    custom_vjp=config.adjoint,
                    reortho_vjp=forward_reortho,
                )

                @jax.jit
                def decompose_reconstruct(x):
                    a = unflatten(x)

                    result = hess_func(matvec_sym, (height, width), v_aug, a)
                    rlrlrl = result.Q_tall
                    ls = rlrlrl[:height, 1::2]
                    rs = rlrlrl[height:, 0::2]
                    ababab = (jnp.diag(result.J_small, k=1) + jnp.diag(result.J_small, k=-1)) / 2
                    alphas = ababab[::2]
                    betas = ababab[1::2]

                    B = jnp.diag(alphas) + jnp.diag(betas, k=1)
                    return jax.flatten_util.ravel_pytree(ls @ B @ rs.T)[0]

            eye = jnp.eye(len(flat), dtype=dtype)

            if configuration[0] == "rec":
                decomposition_diff = flat - decompose_reconstruct(flat)
                reconstruct_error = jnp.sqrt(jnp.mean(decomposition_diff**2))
                reconstruct_loss.append(reconstruct_error.item())

            elif configuration[0] == "jac":
                jacobian_diff = eye - jax.jacrev(decompose_reconstruct)(flat)
                jacobian_error = jnp.sqrt(jnp.mean(jacobian_diff**2))
                jacobian_loss.append(jacobian_error.item())

        if configuration[0] == "jac":
            print("\n Jacobian loss:", jacobian_loss)
        else:
            print("\n Reconstruction loss:", reconstruct_loss)

        line_style = dict(styles[config])
        if style_overrides and config in style_overrides:
            line_style.update(style_overrides[config])

        match configuration[0]:
            case "jac":
                ax.semilogy(
                    ns,
                    jnp.asarray(jacobian_loss),
                    label=label + (" (NAN!)" if jnp.any(jnp.isnan(jnp.asarray(jacobian_loss))) else ""),
                    linewidth=1.0,
                    alpha=0.8,
                    **line_style,
                )
            case "rec":
                ax.semilogy(
                    ns,
                    jnp.asarray(reconstruct_loss),
                    label=label + (" (NAN!)" if jnp.any(jnp.isnan(jnp.asarray(reconstruct_loss))) else ""),
                    **line_style,
                )

        ax.legend(fontsize="small", loc="center right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


jac_configs = (
    "jac",
    {
        bidi_backprop: "Autodiff (w/o reortho)",
        bidi_backprop_reo: "Autodiff (w/ reortho)",
        bidi_adjoint_reo: "Adjoint (w/o reproj)",
        bidi_adjoint_rep: "Adjoint (w/ reproj)",
    },
)

hilbert_jac_configs = (
    "jac",
    {
        bidi_backprop: "Autodiff (w/o reortho)",
        bidi_backprop_reo: "Autodiff (w/ reortho)",
        bidi_adjoint_reo: "Adjoint (w/o reproj)",
        bidi_adjoint_rep: "Adjoint (w/ reproj)",
    },
)

hilbert_rec_configs = (
    "rec",
    {
        bidi_backprop: "Forward (w/o reortho)",
        bidi_backprop_reo: "Forward (w/ reortho)",
    },
)

reproj_configs = (
    "jac",
    {
        bidi_adjoint_rep: "BD: Adj. w/ re-proj.",
        bidi_adjoint_reo: "BD: Adj. w/o re-proj.",
    },
)

rec_configs = (
    "rec",
    {
        bidi_backprop: "Bidiag (w/o reortho)",
        bidi_backprop_reo: "Bidiag (w/  reortho)",
        hess_backprop: "Hess (w/o reortho)",
        hess_backprop_reo: "Hess (w/ reortho)",
    },
)


styles = {
    bidi_backprop: {"color": "C0"},
    bidi_backprop_reo: {"color": "C1"},
    bidi_adjoint_reo: {"color": "C2"},
    bidi_adjoint_rep: {"color": "C3"},
    bidi_adjoint_rep2: {"color": "C3", "linestyle": "--"},
    hess_backprop: {"color": "C4"},
    hess_backprop_reo: {"color": "C5"},
}

reproj_style_overrides = {
    bidi_adjoint_rep: {"color": "C2", "linestyle": "--", "zorder": 100},
    bidi_adjoint_reo: {"color": "C3", "linestyle": "-"},
}


def figures_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "latex", "figures")
    os.makedirs(out, exist_ok=True)
    return out


def make_stability_figure():
    """Gradient error vs matrix size (normal matrices, autodiff vs adjoint)."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        return run_experiment(
            jac_configs,
            ns=jnp.arange(4, 40, step=3),
            matrix_fn=normal_matrix,
            xlabel="Matrix size",
            ylabel="Gradient error",
            dtype=jnp.float64,
        )
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def make_hilbert_stability_figure():
    """Gradient error vs matrix size on square Cauchy matrices.

    Dense n x n Cauchy matrices C_ij = 1/(1+|i-j|), Krylov depth k = n,
    float32. Milder conditioning than the Hilbert matrix (kappa ~ O(n)).
    """
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        return run_experiment(
            hilbert_jac_configs,
            ns=jnp.arange(4, 41, step=2),
            matrix_fn=cauchy_matrix,
            xlabel="Matrix size $n$",
            ylabel="Gradient error",
            dtype=jnp.float32,
        )
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def make_hilbert_reconstruction_figure():
    """Forward reconstruction error on square Cauchy matrices.

    RMS error ||A - LBR^T|| for bidiagonalization with and without forward
    reorthonormalization (two passes). float32, k = n.
    """
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        return run_experiment(
            hilbert_rec_configs,
            ns=jnp.arange(4, 41, step=2),
            matrix_fn=cauchy_matrix,
            xlabel="Matrix size $n$",
            ylabel="Reconstruction error",
            dtype=jnp.float32,
        )
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def make_reprojection_figure():
    """Adjoint gradient error with vs without reprojection.

    Matches the Aug 2025 identity-Jacobian experiment: dense random normal
    matrices of shape n x (n/2), Krylov depth k = n/2, float32.
    """
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        return run_experiment(
            reproj_configs,
            ns=jnp.arange(32, 121, step=8),
            matrix_fn=normal_matrix,
            width_fn=lambda n: n // 2,
            num_matvecs_fn=lambda _n, width: width,
            xlabel="Matrix size $n$",
            ylabel="Loss of accuracy",
            style_overrides=reproj_style_overrides,
            dtype=jnp.float32,
        )
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


if __name__ == "__main__":
    out = figures_dir()

    fig = make_stability_figure()
    fig.savefig(os.path.join(out, "fig_stability.pdf"), bbox_inches="tight")
    print("SAVED_STABILITY_FIGURE")

    fig = make_reprojection_figure()
    fig.savefig(os.path.join(out, "fig_reprojection.pdf"), bbox_inches="tight")
    print("SAVED_REPROJECTION_FIGURE")

    fig = make_hilbert_stability_figure()
    fig.savefig(os.path.join(out, "fig_hilbert_stability.pdf"), bbox_inches="tight")
    print("SAVED_HILBERT_STABILITY_FIGURE")

    fig = make_hilbert_reconstruction_figure()
    fig.savefig(os.path.join(out, "fig_hilbert_reconstruction.pdf"), bbox_inches="tight")
    print("SAVED_HILBERT_RECONSTRUCTION_FIGURE")
