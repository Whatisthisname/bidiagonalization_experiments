#!/usr/bin/env python
from functools import partial
import typing
import numpy as np
import dataclasses
import jax  # type: ignore[import-not-found]
from jax.typing import ArrayLike  # type: ignore[import-not-found]
import jax.numpy as jnp  # type: ignore[import-not-found]

jax.config.update("jax_enable_x64", True)
jnp.printoptions(precision=4)


MatVec = typing.Callable[[ArrayLike], ArrayLike]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class BidiagInput:
    A: ArrayLike
    """(n, m) matrix"""
    v: ArrayLike
    r"""(m,) vector, a.k.a. $\tilde r$"""

    def tree_flatten(self):
        children = (self.A, self.v)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        A, v = children
        return cls(A=A, v=v)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class BidiagOutput:
    rs: ArrayLike
    """(m,k) float array"""
    ls: ArrayLike
    """(n,k) float array"""
    as_: ArrayLike
    """(k,) float array"""
    bs: ArrayLike
    """(k-1,) float array"""
    c: float
    """(k-1,) float array"""
    res: ArrayLike
    """(m,) vector, beta_k * r_{k+1}"""

    def tree_flatten(self):
        children = (
            self.rs,
            self.ls,
            self.as_,
            self.bs,
            self.c,
            self.res,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def L(self) -> ArrayLike:
        """(n, k) float array"""
        return self.ls

    @property
    def B(self) -> ArrayLike:
        """(k, k) float array"""
        as_diag = jnp.diag(self.as_)
        for i in range(len(self.bs)):
            as_diag = as_diag.at[i, i + 1].set(self.bs[i])
        return as_diag

    @property
    def R(self) -> ArrayLike:
        """(m, k) float array"""
        return self.rs

    @property
    def iterations_finished(self) -> int:
        """Accurate only up to floating point precision..."""

        # returns index of first zero element or the last element if no zero element is found
        def first_zero_or_len(arr):
            zeros = jnp.isclose(arr, 0.0, atol=1e-6)
            return jnp.where(jnp.any(zeros), jnp.argmax(zeros), len(arr) - 1)

        return first_zero_or_len(self.as_)


def __bidiagonalize_matvec(num_matvecs: int, reorthogonalize: bool = True):
    def bidiagonalize_matvec(
        matvec: MatVec,
        r1_tilde: ArrayLike,
        *matvec_params,
    ) -> BidiagOutput:
        _, vecmat_fun = jax.vjp(matvec, r1_tilde, *matvec_params)

        def vecmat(v):
            return vecmat_fun(v)[0]

        (ncols,) = np.shape(r1_tilde)
        w0_like = jax.eval_shape(matvec, r1_tilde, *matvec_params)
        (nrows,) = np.shape(w0_like)

        c = 1 / jnp.linalg.norm(r1_tilde)

        k = num_matvecs

        as_ = jnp.zeros((k))
        bs = jnp.zeros((k))
        rs = jnp.zeros((ncols, k + 1))
        rs = rs.at[:, 0].set(r1_tilde * c)
        ls = jnp.zeros((nrows, k))

        CarryState = typing.NamedTuple(
            "CarryState",
            [
                ("rs", ArrayLike),
                ("ls", ArrayLike),
                ("as_", ArrayLike),
                ("bs", ArrayLike),
            ],
        )

        def body_fun(i, carry: CarryState) -> CarryState:
            # n = i + 1
            n = i
            # Forward pass step
            if True:
                t = (
                    matvec(carry.rs[:, n], *matvec_params)
                    - carry.bs[n - 1] * carry.ls[:, n - 1]
                )

                new_alpha, new_l = jax.lax.cond(
                    pred=jnp.allclose(t, 0, atol=1e-6),  # | jnp.isnan(alpha_k),
                    true_fun=lambda: (0.0, jnp.zeros_like(t)),
                    false_fun=lambda: (jnp.linalg.norm(t), t / jnp.linalg.norm(t)),
                )

                as_ = carry.as_.at[n].set(new_alpha)

                if reorthogonalize:
                    mask = jnp.triu(jnp.ones((k + 1, k + 1)), k=1)
                    masked_rs = mask[:, n][None, :] * carry.rs
                    # jax.debug.print("censored rs: \n{}", censored_rs.round(1))
                    rs = carry.rs.at[:, n].set(
                        carry.rs[:, n] - masked_rs @ masked_rs.T @ carry.rs[:, n]
                    )
                    ls = carry.ls.at[:, n].set(new_l - carry.ls @ carry.ls.T @ new_l)
                else:
                    ls = carry.ls.at[:, n].set(new_l)
                    rs = carry.rs

                w = vecmat(ls[:, n]) - as_[n] * rs[:, n]
                # beta_k = jnp.linalg.norm(w)

                new_beta, new_r = jax.lax.cond(
                    pred=jnp.allclose(w, 0, atol=1e-6),  # | jnp.isnan(beta_k),
                    true_fun=lambda: (0.0, jnp.zeros_like(w)),
                    false_fun=lambda: (jnp.linalg.norm(w), w / jnp.linalg.norm(w)),
                )

                bs = carry.bs.at[n].set(new_beta)
                rs = rs.at[:, n + 1].set(new_r)

            return CarryState(
                rs=rs,
                ls=ls,
                as_=as_,
                bs=bs,
            )

        # Run the loop
        loop_out = jax.lax.fori_loop(
            lower=0,
            upper=num_matvecs,
            body_fun=body_fun,
            init_val=CarryState(
                rs=rs,
                ls=ls,
                as_=as_,
                bs=bs,
            ),
        )

        # Create primal output
        primal_output = BidiagOutput(
            c=c,
            res=loop_out.bs[-1] * loop_out.rs[:, -1],
            rs=loop_out.rs[:, :-1],
            ls=loop_out.ls,
            as_=loop_out.as_,
            bs=loop_out.bs[:-1],
        )
        return primal_output

    return bidiagonalize_matvec


@partial(jax.jit, static_argnames=("num_matvecs",))
def bidiagonalize_jvp(
    primals: tuple[ArrayLike, ArrayLike],
    tangents: tuple[ArrayLike, ArrayLike],
    num_matvecs: int,
) -> tuple[BidiagOutput, BidiagOutput]:
    v, A = primals
    d_v, dA = tangents

    c = 1 / jnp.linalg.norm(v)

    size = num_matvecs + 1

    as_ = jnp.zeros((size))
    bs = jnp.zeros((size))
    rs = jnp.zeros((A.shape[1], size + 1))
    rs = rs.at[:, 1].set(v * c)
    ls = jnp.zeros((A.shape[0], size))

    # Initialize tangent variables
    d_as = jnp.zeros((size))
    d_bs = jnp.zeros((size))
    d_rs = jnp.zeros((A.shape[1], size + 1))
    d_rs = d_rs.at[:, 1].set((d_v - v * (v.T @ d_v) / (v @ v)) / jnp.linalg.norm(v))
    d_ls = jnp.zeros((A.shape[0], size))
    d_res = jnp.zeros((A.shape[0]))

    CarryState = typing.NamedTuple(
        "CarryState",
        [
            ("rs", ArrayLike),
            ("d_rs", ArrayLike),
            ("ls", ArrayLike),
            ("d_ls", ArrayLike),
            ("as_", ArrayLike),
            ("d_as", ArrayLike),
            ("bs", ArrayLike),
            ("d_bs", ArrayLike),
        ],
    )

    def body_fun(i, carry: CarryState) -> CarryState:
        n = i + 1

        # Forward pass step
        if True:
            t = A @ carry.rs[:, n] - carry.bs[n - 1] * carry.ls[:, n - 1]

            new_alpha, new_l = jax.lax.cond(
                pred=jnp.allclose(t.T @ t, 0, atol=1e-6),  # | jnp.isnan(alpha_k),
                true_fun=lambda: (0.0, jnp.zeros_like(t)),
                false_fun=lambda: (jnp.linalg.norm(t), t / jnp.linalg.norm(t)),
            )

            as_ = carry.as_.at[n].set(new_alpha)
            ls = carry.ls.at[:, n].set(new_l)

            w = A.T @ ls[:, n] - as_[n] * carry.rs[:, n]

            new_beta, new_r = jax.lax.cond(
                pred=jnp.allclose(w.T @ w, 0, atol=1e-6),  # | jnp.isnan(beta_k),
                true_fun=lambda: (0.0, jnp.zeros_like(w)),
                false_fun=lambda: (jnp.linalg.norm(w), w / jnp.linalg.norm(w)),
            )

            bs = carry.bs.at[n].set(new_beta)
            rs = carry.rs.at[:, n + 1].set(new_r)

        # Tangent map stuff
        if True:
            d_t = (
                dA @ rs[:, n]
                + A @ carry.d_rs[:, n]
                - carry.d_bs[n - 1] * ls[:, n - 1]
                - bs[n - 1] * carry.d_ls[:, n - 1]
            )
            d_alpha_n = ls[:, n].T @ d_t
            d_as = carry.d_as.at[n].set(d_alpha_n)

            d_l_n = jax.lax.cond(
                pred=jnp.allclose(new_alpha, 0, atol=1e-6) | jnp.isnan(new_alpha),
                true_fun=lambda: jnp.zeros_like(ls[:, 0]),
                false_fun=lambda: (d_t - d_alpha_n * ls[:, n]) / new_alpha,
            )
            d_ls = carry.d_ls.at[:, n].set(d_l_n)

            d_w = (
                dA.T @ ls[:, n]
                + A.T @ d_ls[:, n]
                - d_as[n] * rs[:, n]
                - as_[n] * carry.d_rs[:, n]
            )
            d_beta_n = rs[:, n + 1].T @ d_w
            d_bs = carry.d_bs.at[n].set(d_beta_n)

            d_r_np1 = jax.lax.cond(
                pred=jnp.allclose(new_beta, 0, atol=1e-6) | jnp.isnan(new_beta),
                true_fun=lambda: jnp.zeros_like(rs[:, 0]),
                false_fun=lambda: (d_w - d_beta_n * rs[:, n + 1]) / new_beta,
            )
            d_rs = carry.d_rs.at[:, n + 1].set(d_r_np1)

        return CarryState(
            rs=rs,
            d_rs=d_rs,
            ls=ls,
            d_ls=d_ls,
            as_=as_,
            d_as=d_as,
            bs=bs,
            d_bs=d_bs,
        )

    # Run the loop
    loop_out = jax.lax.fori_loop(
        lower=0,
        upper=num_matvecs,
        body_fun=body_fun,
        init_val=CarryState(
            rs=rs,
            d_rs=d_rs,
            ls=ls,
            d_ls=d_ls,
            as_=as_,
            d_as=d_as,
            bs=bs,
            d_bs=d_bs,
        ),
    )

    # Compute d_c
    d_c = -(v @ d_v) / (v @ v * jnp.linalg.norm(v))

    k = num_matvecs

    d_res = (
        A.T @ loop_out.d_ls[:, k]
        + dA.T @ loop_out.ls[:, k]
        - loop_out.as_[k] * loop_out.d_rs[:, k]
        - loop_out.d_as[k] * loop_out.rs[:, k]
    )

    # Create primal output
    primal_output = BidiagOutput(
        c=c,
        res=loop_out.bs[k] * loop_out.rs[:, k + 1],
        rs=loop_out.rs[:, 1:-1],
        ls=loop_out.ls[:, 1:],
        as_=loop_out.as_[1:],
        bs=loop_out.bs[1:-1],
    )

    # Create tangent output
    tangent_output = BidiagOutput(
        c=d_c,
        res=d_res,
        rs=loop_out.d_rs[:, 1:-1],
        ls=loop_out.d_ls[:, 1:],
        as_=loop_out.d_as[1:],
        bs=loop_out.d_bs[1:-1],
    )

    return primal_output, tangent_output


BidiagCache = typing.NamedTuple(
    "BidiagCache",
    [
        ("primal", BidiagOutput),
        ("A", ArrayLike),
        ("v", ArrayLike),
    ],
)

BidiagCache_matvec = typing.NamedTuple(
    "BidiagCache_matvec",
    [
        ("primal", BidiagOutput),
        ("v", ArrayLike),
    ],
)


CarryState = typing.NamedTuple(
    "CarryState",
    [
        ("up_i_p_1", ArrayLike),
        ("down_i", ArrayLike),
        ("param_incremental_grads", ArrayLike),
        ("Sigma", ArrayLike),
        ("Omega", ArrayLike),
    ],
)


def bidiagonalize(
    num_matvecs: int,
    custom_vjp: bool = True,
    reorthogonalize: bool = True,
):
    primal_map = __bidiagonalize_matvec(
        num_matvecs=num_matvecs, reorthogonalize=reorthogonalize
    )

    def _bidiag_vjp_fwd(
        matvec: MatVec, v0: ArrayLike, *matvec_params
    ) -> tuple[BidiagOutput, tuple[BidiagCache_matvec, tuple]]:
        matvec_convert, aux_args = jax.closure_convert(
            lambda u, *v: matvec(u, *v), v0, *matvec_params
        )

        primal = primal_map(matvec_convert, v0, *matvec_params, *aux_args)
        cache = BidiagCache_matvec(
            primal=primal,
            v=v0,
        )
        return primal, (cache, matvec_params)

    def _bidiag_vjp_bwd(
        matvec: MatVec,
        cache_and_params: tuple[BidiagCache_matvec, tuple],
        d: BidiagOutput,
    ) -> BidiagInput:
        cache, matvec_params = cache_and_params
        _, vecmat_fun = jax.vjp(lambda v, p: matvec(v, *p), cache.v, matvec_params)

        def vecmat(v):
            return vecmat_fun(v)[0]

        w0_like = jax.eval_shape(matvec, cache.v, *matvec_params)
        (n,) = np.shape(w0_like)
        (m,) = np.shape(cache.v)

        # Unpack primal variables from cache. These are 0-indexed.
        rs = cache.primal.rs
        ls = cache.primal.ls
        bs = cache.primal.bs
        bs = jnp.append(bs, jnp.array([-1.0]))
        as_ = cache.primal.as_
        res = cache.primal.res
        c = cache.primal.c

        del cache

        drs = d.rs
        dls = d.ls
        dbs = d.bs
        dbs = jnp.append(dbs, jnp.array([0.0]))
        das = d.as_
        das = jnp.append(das, jnp.array([0.0]))
        dres = d.res
        dc = d.c

        k = num_matvecs

        assert num_matvecs >= 1

        gamma = -rs.T @ dres
        del d  # so we don't accidentally use it later.

        gamma = gamma.at[-1].add(das[k - 1])
        down_k = dres + rs @ gamma

        just_in_case = c * jnp.array([dc])

        upper_tri = jnp.triu(jnp.ones((k, k + 1)))

        def body_fun(i_in: int, carry: CarryState):
            # 'i_in' will go from 0 to k-1 (inclusive)
            i = k - 1 - i_in
            # so 'i' will go from k-1 to 0 (inclusive)

            # Reortho the "down" contained in the carry
            down_i = carry.down_i
            if reorthogonalize:
                correction = jnp.zeros(shape=k).at[i].set(das[i])
                down_i = (
                    down_i
                    - rs @ (upper_tri[:, i + 1] * (rs.T @ down_i))
                    + rs @ correction
                )

            A_down_i = matvec(carry.down_i, *matvec_params)
            Sigma_SigmaT = -ls.T @ (dls[:, i] + A_down_i)
            Sigma_SigmaT = Sigma_SigmaT.at[i - 1].add(as_[i] * dbs[i - 1])

            Sigma_SigmaT = Sigma_SigmaT.at[i].add(bs[i] * dbs[i])
            Sigma = carry.Sigma.at[i, :].add(Sigma_SigmaT * upper_tri[:, i])
            Sigma = Sigma.at[i, i].divide(2.0)

            up_i = (
                dls[:, i]
                + A_down_i
                + ls @ (Sigma + Sigma.T)[:, i]
                - carry.up_i_p_1 * bs[i]
            )
            up_i /= as_[i]

            # Reortho the "up" we have just produced
            if reorthogonalize:
                correction = jnp.zeros(shape=k).at[i - 1].set(dbs[i - 1])
                up_i = up_i - ls @ (upper_tri[:, i] * (ls.T @ up_i)) + ls @ correction

            # Second phase
            AT_up_i = vecmat(up_i)
            Omega_OmegaT = -rs.T @ (drs[:, i] + AT_up_i)
            Omega_OmegaT = Omega_OmegaT.at[i - 1].add(das[i - 1] * bs[i - 1])
            Omega_OmegaT = Omega_OmegaT.at[i].add(
                as_[i] * das[i] - just_in_case.at[i].get(mode="fill", fill_value=0.0)
            )
            Omega = carry.Omega.at[i, :].add(Omega_OmegaT * upper_tri[:, i])
            Omega = Omega.at[i, i].divide(2.0)

            downs_i_m_1 = (
                drs[:, i]
                + AT_up_i
                + rs @ (Omega + Omega.T)[:, i]
                - as_[i] * down_i
                + res * gamma[i]
            )
            downs_i_m_1 /= bs[i - 1]

            def parameter_gradient_getter(params, up, r, l, down):
                return up @ matvec(r, *params) + l @ matvec(down, *params)

            new_param_grad_incr = jax.grad(parameter_gradient_getter, argnums=0)(
                matvec_params,
                up_i,
                rs[:, i],
                ls[:, i],
                carry.down_i,
            )

            return CarryState(
                up_i_p_1=up_i,
                down_i=downs_i_m_1,
                param_incremental_grads=jax.tree_util.tree_map(
                    lambda running_sum, grad_component: running_sum + grad_component,
                    carry.param_incremental_grads,
                    new_param_grad_incr,
                ),
                Omega=Omega,
                Sigma=Sigma,
            )

        output: CarryState = jax.lax.fori_loop(
            lower=0,
            upper=k,
            body_fun=body_fun,
            init_val=CarryState(
                up_i_p_1=jnp.zeros(n),
                down_i=down_k,
                param_incremental_grads=jax.tree.map(jnp.zeros_like, matvec_params),
                Sigma=jnp.zeros((k, k)),
                Omega=jnp.zeros((k, k)),
            ),
        )

        kappa = output.down_i
        param_grads_out = output.param_incremental_grads

        return (
            -c * kappa,
            *param_grads_out,
        )

    if custom_vjp:
        _bidiagonalize = jax.custom_vjp(
            primal_map,
            nondiff_argnums=(0,),
        )
        _bidiagonalize.defvjp(
            _bidiag_vjp_fwd,
            _bidiag_vjp_bwd,
        )
    else:
        _bidiagonalize = primal_map
    return _bidiagonalize
