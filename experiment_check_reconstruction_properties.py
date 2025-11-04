import dataclasses
from typing import Literal
from algo_bidiag import bidiagonalize as bidiagonalize

import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from tueplots import axes, figsizes, fontsizes
import os
from algo_hessenberg import hessenberg


def hilbert_matrix(ndim, /):
    a = jnp.arange(ndim)
    return 1 / (1 + (a[:, None] + a[None, :]))


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


hess_backprop = Config(algo="Hess", adjoint=False, reortho=False)
bidi_backprop = Config(algo="Bidi", adjoint=False, reortho=False)
hess_backprop_reo = Config(algo="Hess", adjoint=False, reortho=True)
bidi_backprop_reo = Config(algo="Bidi", adjoint=False, reortho=True)
hess_adjoint_reo = Config(algo="Hess", adjoint=True, reortho=True, reproj_adj="none")
bidi_adjoint_reo = Config(algo="Bidi", adjoint=True, reortho=True, reproj_adj="none")
hess_adjoint_rep = Config(algo="Hess", adjoint=True, reortho=True, reproj_adj="match")
bidi_adjoint_rep = Config(algo="Bidi", adjoint=True, reortho=True, reproj_adj="match")


def run_experiment(configuration) -> None:
    fig, (ax) = plt.subplots(figsize=(10, 10))

    for config, label in tqdm.tqdm(configuration[1].items(), desc="Testing setups"):
        ns = jnp.arange(4, 32, step=8)
        reconstruct_loss = []
        jacobian_loss = []

        for n in tqdm.tqdm(ns, desc=f"Testing {label}", leave=False):
            height = int(n)
            width = height  # height // 2  #  + height // 4
            A = hilbert_matrix(height)[:, :width]
            A = jax.random.normal(key_A, shape=(n, width))
            v = jax.random.normal(key_v, shape=(width,))
            flat, unflatten = jax.flatten_util.ravel_pytree(A)

            if config.algo == "Bidi":
                bd_func = bidiagonalize(
                    num_matvecs=width,
                    custom_vjp=config.adjoint,
                    reorthogonalize=config.reortho,
                    also_reorthogonalize_vjp=(config.reproj_adj == "match"),
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

                v_aug = jnp.concat([jnp.zeros(height), v])

                hess_func = hessenberg(
                    height + width,
                    reortho=config.reortho,
                    custom_vjp=config.adjoint,
                    reortho_vjp=config.reproj_adj,
                )

                @jax.jit
                def decompose_reconstruct(x):
                    a = unflatten(x)

                    result = hess_func(matvec_sym, (n, width), v_aug, a)
                    rlrlrl = result.Q_tall
                    ls = rlrlrl[:n, 1::2]
                    rs = rlrlrl[n:, 0::2]
                    ababab = (
                        jnp.diag(result.J_small, k=1) + jnp.diag(result.J_small, k=-1)
                    ) / 2
                    alphas = ababab[::2]
                    betas = ababab[1::2]
                    # res = result.residual[height:]

                    B = jnp.diag(alphas) + jnp.diag(betas, k=1)
                    return jax.flatten_util.ravel_pytree(ls @ B @ rs.T)[0]

            if configuration[0] == "rec":
                # The norm of the input vs the output should be zero if it faithfully reconstructs.
                decomposition_diff = flat - decompose_reconstruct(flat)
                reconstruct_error = jnp.sqrt(jnp.mean(decomposition_diff**2))
                reconstruct_loss.append(reconstruct_error.item())

            elif configuration[0] == "jac":
                # The jacobian of the identity function should be a diagonal matrix with all 1s - the identity matrix.
                jacobian_diff = jnp.eye(len(flat)) - jax.jacrev(decompose_reconstruct)(
                    flat
                )
                jacobian_error = jnp.sqrt(jnp.mean(jacobian_diff**2))
                jacobian_loss.append(jacobian_error.item())

        print("\n Loss is:", jacobian_loss)
        match configuration[0]:
            case "jac":
                ax.semilogy(
                    ns,
                    jnp.asarray(jacobian_loss),
                    label=label
                    + (
                        " (NAN!)"
                        if jnp.any(jnp.isnan(jnp.asarray(jacobian_loss)))
                        else ""
                    ),
                    linewidth=1.0,
                    alpha=0.8,
                    **styles[config],
                )
            case "rec":
                ax.semilogy(
                    ns,
                    jnp.asarray(reconstruct_loss),
                    label=label
                    + (
                        " (NAN!)"
                        if jnp.any(jnp.isnan(jnp.asarray(reconstruct_loss)))
                        else ""
                    ),
                    **styles[config],
                )

        ax.legend(fontsize="xx-small")
        ax.set_xlabel("Matrix size", fontsize="small")
        ax.set_ylabel("Accuracy", fontsize="small")
        ax.grid(True, alpha=0.3)


jac_configs = (
    "jac",
    {
        bidi_backprop: "Bidiag: Backprop (w/o reortho)",
        bidi_backprop_reo: "Bidiag: Backprop (w/  reortho)",
        bidi_adjoint_reo: "Bidiag: Adjoint (w reortho, w/o reproj)",
        bidi_adjoint_rep: "Bidiag: Adjoint (w reortho, w reproj)",
    },
)

rec_configs = (
    "rec",
    {
        bidi_backprop: "Bidiag (w/o reortho)",
        bidi_backprop_reo: "Bidiag (w/  reortho)",
        hess_backprop: "Hess (w/o reortho)",
        hess_backprop_reo: "Hess (w/ reortho)",
        # TODO add double reortho as well
    },
)


styles = {
    bidi_backprop: {"color": "C0"},
    bidi_backprop_reo: {"color": "C1"},
    bidi_adjoint_reo: {"color": "C2"},
    bidi_adjoint_rep: {"color": "C3"},
    hess_backprop: {"color": "C4"},
    hess_backprop_reo: {"color": "C5"},
}

if __name__ == "__main__":
    run_experiment(jac_configs)


def matching_directory(file, where, /, replace="experiments/"):
    if where not in ["data/", "figures/", "results/"]:
        raise ValueError
    if replace not in ["experiments/"]:
        raise ValueError

    # Read directory name and replace "experiments" with e.g. "data"
    directory_file = os.path.dirname(file) + "/"
    return directory_file.replace(replace, where)


directory_fig = matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
plt.savefig(f"{directory_fig}accuracy_loss.pdf")

plt.show()
