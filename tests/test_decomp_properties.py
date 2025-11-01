import jax
import jax.numpy as jnp
import os
import sys
import operator as op
import pytest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import algo_bidiag

jax.config.update("jax_enable_x64", True)

_current_tolerance = float(os.environ.get("ATOL", "1e-8"))


def very_close(a, b, atol=None):
    return jnp.allclose(a, b, atol=atol if atol else _current_tolerance)


def _check_properties_of_estimator(*, estimate, n, m):
    for i in range(10):
        A = jax.random.normal(jax.random.PRNGKey(i), shape=(n, m))
        v = jax.random.normal(jax.random.PRNGKey(100 * i), shape=(m))
        result: algo_bidiag.BidiagOutput = estimate(lambda vv, pp: pp @ vv, v, A)

        primal_evaluations = {
            "left ortho      ": _eval_orthogonality(result.L),
            "right ortho     ": _eval_orthogonality(result.R),
            "residual ortho  ": _eval_residual_orthogonal(result),
            "bidiagonalizes A": _eval_bidiagonalizes(A, result),
            # "always True     ": (True, 0.0),
        }

        if not all(map(op.itemgetter(0), primal_evaluations.values())):
            msg = "\n".join(
                f"{k}: {str(eval):5}  err:{measure}"
                for k, (eval, measure) in primal_evaluations.items()
            )
            raise ValueError(f"One or more properties did not hold: \n{msg}")


def _eval_orthogonality(vs) -> tuple[bool, float]:
    inner = vs.T @ vs
    evaluation = very_close(inner, jnp.eye(inner.shape[0]))
    measurement = jnp.linalg.norm(inner - jnp.eye(inner.shape[0]))
    return evaluation, measurement


def _eval_bidiagonalizes(A, result: algo_bidiag.BidiagOutput) -> tuple[bool, float]:
    evaluation = very_close(result.B, result.L.T @ A @ result.R)
    measurement = jnp.linalg.norm(result.B - result.L.T @ A @ result.R)
    return evaluation, measurement


def _eval_residual_orthogonal(result: algo_bidiag.BidiagOutput) -> tuple[bool, float]:
    evaluation = very_close(result.R.T @ result.res, jnp.zeros(result.matvec_num))
    measurement = jnp.linalg.norm(result.R.T @ result.res)
    return evaluation, measurement


def test_properties_no_reortho():
    with pytest.raises(ValueError):  # will raise because we need reortho.
        reortho = False
        estimate = algo_bidiag.bidiagonalize(num_matvecs=3, reorthogonalize=reortho)
        _check_properties_of_estimator(estimate=estimate, n=5, m=4)
        estimate = algo_bidiag.bidiagonalize(num_matvecs=19, reorthogonalize=reortho)
        _check_properties_of_estimator(estimate=estimate, n=20, m=25)
        estimate = algo_bidiag.bidiagonalize(num_matvecs=50, reorthogonalize=reortho)
        _check_properties_of_estimator(estimate=estimate, n=100, m=150)


def test_properties_reortho():
    reortho = True
    estimate = algo_bidiag.bidiagonalize(num_matvecs=3, reorthogonalize=reortho)
    _check_properties_of_estimator(estimate=estimate, n=5, m=4)
    estimate = algo_bidiag.bidiagonalize(num_matvecs=19, reorthogonalize=reortho)
    _check_properties_of_estimator(estimate=estimate, n=20, m=25)
    estimate = algo_bidiag.bidiagonalize(num_matvecs=50, reorthogonalize=reortho)
    _check_properties_of_estimator(estimate=estimate, n=100, m=150)


def test_custom_vjp():
    return
    estimate_custom = algo_bidiag.bidiagonalize(
        num_matvecs=3, reorthogonalize=False, custom_vjp=True
    )
    estimate = algo_bidiag.bidiagonalize(
        num_matvecs=3, reorthogonalize=False, custom_vjp=False
    )
    n, m = 4, 3
    for i in range(10):
        A = jax.random.normal(jax.random.PRNGKey(i), shape=(n, m))
        v = jax.random.normal(jax.random.PRNGKey(100 * i), shape=(m))

        def matvec(v, A):
            return A @ v

        result: algo_bidiag.BidiagOutput = estimate(matvec, v, A)
        print("result")
        print(result)

        _, vjpfun_custom = jax.vjp(lambda v, p: estimate_custom(matvec, v, p), v, A)
        v_grad_c, param_grad_c = vjpfun_custom(result)

        _, vjpfun_autodiff = jax.vjp(lambda v, p: estimate(matvec, v, p), v, A)
        v_grad, param_grad = vjpfun_autodiff(result)

        print("v grads:")
        print(v_grad)
        print(v_grad_c)
        print("param grads:")
        print(param_grad)
        print(param_grad_c)

        adjoint_evaluations = {
            "param grad match": (
                very_close(param_grad_c, param_grad),
                jnp.linalg.norm(param_grad_c - param_grad),
            ),
            "v gradient match": (
                very_close(v_grad_c, v_grad),
                jnp.linalg.norm(v_grad_c - v_grad),
            ),
        }

        if not all(map(op.itemgetter(0), adjoint_evaluations.values())):
            msg = "\n".join(
                f"{k}: {str(eval):5}  err:{measure}"
                for k, (eval, measure) in adjoint_evaluations.items()
            )
            raise ValueError(f"One or more properties did not hold: \n{msg}")
