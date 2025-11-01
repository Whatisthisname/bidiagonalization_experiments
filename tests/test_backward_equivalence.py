import jax
import jax.numpy as jnp
from matfree import decomp
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algo_hessenberg import hessenberg
from algo_bidiag import bidiagonalize, BidiagOutput


def test_gradient_equivalence():
    jnp.printoptions(precision=None)

    jax.config.update("jax_enable_x64", True)

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    n = 4
    m = 4

    num_matvecs = 4

    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(n, m))

    def matvec(v, *params):
        (A,) = params
        return A @ v

    def matvec_sym(v0, *params):
        (A,) = params
        upper, lower = jnp.split(v0, [n])
        return jnp.concat((A @ lower, A.T @ upper))

    def bidiag_loss(alphas, betas, ls, rs, res, c):
        flattened = jax.flatten_util.ravel_pytree((alphas, betas, ls, rs, res, c))[0]
        return jnp.sum(
            jax.random.normal(jax.random.PRNGKey(0), len(flattened)) * flattened
        )

    def extract_hess_results(result: decomp._DecompResult) -> tuple:
        rlrlrl = result.Q_tall
        ls = rlrlrl[:n, 1::2]
        rs = rlrlrl[n:, 0::2]
        ababab = jnp.diag(result.J_small, k=1)
        alphas = ababab[::2]
        betas = ababab[1::2]
        res = result.residual[n:]
        return alphas, betas, ls, rs, res, result.init_length_inv

    def padded_hess_loss(result: decomp._DecompResult):
        return bidiag_loss(*extract_hess_results(result))

    def extract_bidiag_results(result: BidiagOutput) -> tuple:
        ls, rs = result.ls, result.rs
        res, c = result.res, result.c
        alphas = result.as_
        betas = result.bs
        return alphas, betas, ls, rs, res, c

    def bidiag_materialized_loss(result: BidiagOutput):
        return bidiag_loss(*extract_bidiag_results(result))

    def check_bidiag_properties(alphas, betas, ls, rs, res, c):
        r_inner = rs.T @ rs
        l_inner = ls.T @ ls
        expected = jnp.eye(r_inner.shape[0])
        print("R inner error:", jnp.linalg.norm(r_inner - expected))
        print("L inner error:", jnp.linalg.norm(l_inner - expected))

    v = jax.random.normal(jax.random.PRNGKey(0), shape=(m))
    v_aug = jnp.concat([jnp.zeros(n), v])

    # Look at augmented version:
    hess_func = hessenberg(
        2 * num_matvecs, custom_vjp=True, reortho="full", reortho_vjp="match"
    )
    hess_result = hess_func(matvec_sym, (n, m), v_aug, A)

    hess_loss, hess_grad = jax.value_and_grad(padded_hess_loss)(hess_result)

    _, vjpfun = jax.vjp(lambda v, A: hess_func(matvec_sym, (n, m), v, A), v_aug, A)
    (v_aug_grad, A_aug_grad) = vjpfun(hess_grad)

    # Look at reduced version:
    bd_new_func = bidiagonalize(
        num_matvecs,
        reorthogonalize=True,
        custom_vjp=True,
        also_reorthogonalize_vjp=True,
    )
    bd_new_result: BidiagOutput = bd_new_func(matvec, v, A)

    reduced_hess_loss, reduced_hess_grad = jax.value_and_grad(bidiag_materialized_loss)(
        bd_new_result
    )

    def compare_two_outputs(*args):
        alphas, betas, ls, rs, res, c, _alphas, _betas, _ls, _rs, _res, _c = args
        print("betas:", betas)

        for i, (alpha, l, r, _alpha, _l, _r) in enumerate(
            # for i, (alpha, beta, l, r, _alpha, _beta, _l, _r) in enumerate(
            zip(alphas, ls.T, rs.T, _alphas, _ls.T, _rs.T)
        ):
            print(f"iteration {i}")
            print()
            print("as diff:", jnp.abs(alpha - _alpha))
            # print("bs diff:", jnp.abs(beta - _beta))
            print("r diff:", jnp.abs(r - _r))
            # print("l diff:", jnp.abs(l - _l))
            print()
        print(
            "norm of residuals:",
            jnp.linalg.norm(res),
            "and",
            jnp.linalg.norm(res),
            sep="\n",
        )

    # compare_two_outputs(
    #     *extract_bidiag_results(bd_new_result), *extract_hess_results(hess_result)
    # )

    print(jnp.linalg.norm(reduced_hess_loss - hess_loss))

    assert jnp.allclose(reduced_hess_loss, hess_loss, atol=1e-15, rtol=1e-15)

    _, vjpfun = jax.vjp(lambda v, A: bd_new_func(matvec, v, A), v, A)
    (v_grad_reduced, A_grad_reduced) = vjpfun(reduced_hess_grad)

    # print("hess error norm:")
    # check_bidiag_properties(*extract_hess_results(hess_result))
    # print("reduced error norm:")
    # check_bidiag_properties(*extract_bidiag_results(bd_new_result))

    def assert_close(leaf1, leaf2, name):
        assert jnp.allclose(leaf1, leaf2, rtol=1e-15), (
            f"{name} differ: error magnitude: {jnp.linalg.norm(leaf1 - leaf2)}"
        )

    stuff_names = ("rs", "ls", "alphas", "betas", "res", "c")
    jax.tree.map(
        assert_close,
        extract_hess_results(hess_result),
        extract_bidiag_results(bd_new_result),
        stuff_names,
    )

    grad_names = ("A grad", "v_grad")
    jax.tree.map(
        assert_close,
        (A_aug_grad, v_aug_grad[n:]),
        (A_grad_reduced, v_grad_reduced),
        grad_names,
    )
    # print(A_aug_grad - A_grad_reduced)
