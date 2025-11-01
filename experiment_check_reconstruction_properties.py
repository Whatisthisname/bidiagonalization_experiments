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

plt.subplots(dpi=200)
setups = {
    # (True, False, True, "none"): "AH: Backprop",
    # (False, False, True, "none"): "BD: Backprop",
    # (True, True, True, "match"): "AH: Adj. w/ re-proj.",
    (False, True, True, "match"): "BD: Adj. w/ re-proj.",
    # (True, True, True, "none"): "AH: Adj. w/o re-proj.",
    (False, True, True, "none"): "BD: Adj. w/o re-proj.",
}
loss_type: Literal["reconstruct", "jacobian"] = "reconstruct"
styles_all = {"linewidth": 1.0, "alpha": 0.8}
styles = {
    (True, True, True, "match"): {
        "linestyle": "dashed",
        "zorder": 100,
        "color": "C0",
        **styles_all,
    },
    (True, True, True, "none"): {"linestyle": "solid", "color": "C1", **styles_all},
    (True, False, True, "none"): {"linewidth": 4, "color": "gray", "alpha": 0.35},
    (False, True, True, "match"): {
        "linestyle": "dashed",
        "zorder": 100,
        "color": "C2",
        **styles_all,
    },
    (False, True, True, "none"): {"linestyle": "solid", "color": "C3", **styles_all},
    (False, False, True, "none"): {"linewidth": 4, "color": "black", "alpha": 0.35},
}
for (use_hessenberg, custom, reortho, match), label in tqdm.tqdm(
    setups.items(), desc="Testing setups"
):
    ns = jnp.arange(8, 64, step=8)  # step = 4
    loss = []
    # bd_loss = []
    for n in tqdm.tqdm(ns, desc=f"Testing {label}", leave=False):
        n = int(n)
        # A = hilbert_matrix(n)[:, : n // 2]
        A = jax.random.normal(key_A, shape=(n, n // 2))

        bd_func = bidiagonalize(
            num_matvecs=n // 2,
            custom_vjp=custom,
            reorthogonalize=True,
            also_reorthogonalize_vjp=(match == "match"),
        )

        def matvec_sym(v0, *params):
            (A,) = params
            upper, lower = jnp.split(v0, [n])
            return jnp.concat((A @ lower, A.T @ upper))

        v = jax.random.normal(key_v, shape=(n // 2,))
        v_aug = jnp.concat([jnp.zeros(n), v])

        hess_func = hessenberg(
            n + n // 2, reortho=reortho, custom_vjp=custom, reortho_vjp=match
        )

        flat, unflatten = jax.flatten_util.ravel_pytree(A)

        @jax.jit
        def decompose_reconstruct(x):
            a = unflatten(x)

            if use_hessenberg:
                result = hess_func(matvec_sym, (n, n // 2), v_aug, a)
                rlrlrl = result.Q_tall
                ls = rlrlrl[:n, 1::2]
                rs = rlrlrl[n:, 0::2]
                ababab = (
                    jnp.diag(result.J_small, k=1) + jnp.diag(result.J_small, k=-1)
                ) / 2
                alphas = ababab[::2]
                betas = ababab[1::2]
                res = result.residual[n:]
            else:
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

        # result, vjpfun = jax.vjp(
        #     lambda vv, pp: bd_func(lambda v, p: p @ v, vv, pp), v, A
        # )
        # dv, dA = vjpfun(result)

        if loss_type == "reconstruct":
            # The norm of the input vs the output should be zero if it faithfully reconstructs.
            decomposition_diff = flat - decompose_reconstruct(flat)
            error = jnp.sqrt(jnp.mean(decomposition_diff**2))

        elif loss_type == "jacobian":
            # The jacobian of the identity function should be a diagonal matrix with all 1s - the identity matrix.
            jacobian_diff = jnp.eye(len(flat)) - jax.jacrev(decompose_reconstruct)(flat)
            error = jnp.sqrt(jnp.mean(jacobian_diff**2))

        loss.append(error)

    plt.semilogy(
        ns,
        jnp.asarray(loss),
        label=label,
        **styles[(use_hessenberg, custom, reortho, match)],
    )

plt.legend(fontsize="xx-small")
plt.xlabel("Hilbert matrix size", fontsize="small")
plt.ylabel(f"Loss of\n{loss_type} accuracy", fontsize="small")


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
plt.savefig(f"{directory_fig}{loss_type}_accuracy_loss.pdf")

plt.show()
