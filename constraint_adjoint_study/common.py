"""Shared utilities for the constraint/adjoint study.

Conventions used throughout (stated once, used everywhere):

  x  = (L, R, alphas, betas, res, c)      the solver output ("unknowns")
  th = (A, v)                             the parameters
  con(x, th) = 0                          a constraint set characterizing x(th)

  Adjoint convention:   solve   (d con/d x)^T mu = -grad_x rho
  Gradient formula:     grad_th rho = (d con/d th)^T mu
  (This is Lagrangian stationarity of  Lag = rho + <mu, con>:
   d Lag/dx = 0  and  grad_th = d Lag/d th.)

Dimensions (K = num_matvecs, A is N x M):
  d_U := dim(x)  = NK + MK + K + (K-1) + M + 1 = NK + MK + M + 2K
  d_O := #rows of the ORIGINAL constraint set (paper eqs (1)-(6))
       = MK + NK + M + K(K+1)/2 + K(K+1)/2 + K = NK + MK + M + K^2 + 2K
  d_O - d_U = K^2.
"""

import os
import sys

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo_bidiag import bidiagonalize  # noqa: E402

# ----------------------------------------------------------------------------
# Forward solve and flattening
# ----------------------------------------------------------------------------


def forward(A, v, K, reortho=True):
    """Run the (reorthogonalized) primal and return x as a dict pytree."""
    bd = bidiagonalize(num_matvecs=K, custom_vjp=False, reorthogonalize=reortho)
    out = bd(lambda vec, p: p @ vec, v, A)
    return dict(
        L=out.ls,
        R=out.rs,
        alphas=out.as_,
        betas=out.bs,
        res=out.res,
        c=jnp.asarray(out.c),
    )


def flatteners(N, M, K, dtype=jnp.float64):
    x_like = dict(
        L=jnp.zeros((N, K), dtype),
        R=jnp.zeros((M, K), dtype),
        alphas=jnp.zeros((K,), dtype),
        betas=jnp.zeros((K - 1,), dtype),
        res=jnp.zeros((M,), dtype),
        c=jnp.zeros((), dtype),
    )
    th_like = dict(A=jnp.zeros((N, M), dtype), v=jnp.zeros((M,), dtype))
    _, unflat_x = jax.flatten_util.ravel_pytree(x_like)
    _, unflat_th = jax.flatten_util.ravel_pytree(th_like)
    return unflat_x, unflat_th


def flat(pytree):
    return jax.flatten_util.ravel_pytree(pytree)[0]


def B_of(x):
    return jnp.diag(x["alphas"]) + jnp.diag(x["betas"], k=1)


# ----------------------------------------------------------------------------
# Constraint sets (flat in, flat out)
# ----------------------------------------------------------------------------


def tril_indices(K, offset=0):
    return jnp.tril_indices(K, k=offset)


def con_full(x, th):
    """Original constraint set, paper eqs (1)-(6). d_O rows.

    Row blocks, in order:
      recR : A^T L - R B^T - res e_K^T          (M x K)   multiplier Xi
      recL : A R - L B                          (N x K)   multiplier Phi
      init : R e_1 - c v                        (M,)      multiplier kappa
      orthL: tril(L^T L - I)  (incl diagonal)   (K(K+1)/2) multiplier Sigma
      orthR: tril(R^T R - I)  (incl diagonal)   (K(K+1)/2) multiplier Omega
      orthres: R^T res                          (K,)      multiplier eta
    """
    L, R, res, c = x["L"], x["R"], x["res"], x["c"]
    A, v = th["A"], th["v"]
    B = B_of(x)
    K = L.shape[1]
    eK = jnp.zeros((K,)).at[-1].set(1.0)
    il, jl = jnp.tril_indices(K)
    recR = A.T @ L - R @ B.T - jnp.outer(res, eK)
    recL = A @ R - L @ B
    init = R[:, 0] - c * v
    orthL = (L.T @ L - jnp.eye(K))[il, jl]
    orthR = (R.T @ R - jnp.eye(K))[il, jl]
    orthres = R.T @ res
    return jnp.concatenate(
        [
            recR.ravel(),
            recL.ravel(),
            init,
            orthL,
            orthR,
            orthres,
        ]
    )


def con_red(x, th):
    """Reduced ("unstable primal", appendix B.2) constraint set. d_U rows.

    Row blocks, in order:
      init : r_1 - c v                                    (M,)    mult kappa
      recl : alpha_n l_n - (A r_n - beta_{n-1} l_{n-1})   (N x K) mult lam_n
      recr : beta_n r_{n+1} - (A^T l_n - alpha_n r_n), n<K
             res - (A^T l_K - alpha_K r_K),          n=K  (M x K) mult rho_n
      nrmL : l_n^T l_n - 1                                (K,)    mult sigma_n
      nrmR : r_n^T r_n - 1                                (K,)    mult omega_n
    """
    L, R, al, be, res, c = (x["L"], x["R"], x["alphas"], x["betas"], x["res"], x["c"])
    A, v = th["A"], th["v"]
    N, K = L.shape
    init = R[:, 0] - c * v
    Lprev = jnp.concatenate([jnp.zeros((N, 1)), L[:, :-1]], axis=1)
    beprev = jnp.concatenate([jnp.zeros((1,)), be])  # beta_0 := 0
    recl = L * al[None, :] - (A @ R - Lprev * beprev[None, :])
    Rnext = jnp.concatenate([R[:, 1:], res[:, None]], axis=1)
    benext = jnp.concatenate([be, jnp.ones((1,))])  # coeff of res is 1
    recr = Rnext * benext[None, :] - (A.T @ L - R * al[None, :])
    nrmL = jnp.sum(L * L, axis=0) - 1.0
    nrmR = jnp.sum(R * R, axis=0) - 1.0
    return jnp.concatenate([init, recl.ravel(), recr.ravel(), nrmL, nrmR])


def dims(N, M, K):
    d_U = N * K + M * K + M + 2 * K
    d_O = N * K + M * K + M + K * K + 2 * K
    return d_U, d_O


# ----------------------------------------------------------------------------
# Dense adjoint machinery
# ----------------------------------------------------------------------------


def jacobians(con, x, th, N, M, K):
    """Return (Jx, Jth) of the flat constraint function at (x, th)."""
    unflat_x, unflat_th = flatteners(N, M, K)
    xf, thf = flat(x), flat(th)

    def cf(xf_, thf_):
        return con(unflat_x(xf_), unflat_th(thf_))

    Jx = jax.jacrev(cf, argnums=0)(xf, thf)
    Jth = jax.jacrev(cf, argnums=1)(xf, thf)
    return np.asarray(Jx), np.asarray(Jth)


def dense_adjoint_gradient(con, x, th, grad_x, N, M, K, rcond=None):
    """Solve (Jx^T) mu = -grad_x (lstsq if non-square) and return
    (grad_th, mu, Jx, Jth)."""
    Jx, Jth = jacobians(con, x, th, N, M, K)
    gx = np.asarray(flat(grad_x))
    if Jx.shape[0] == Jx.shape[1]:
        mu = np.linalg.solve(Jx.T, -gx)
    else:
        mu, *_ = np.linalg.lstsq(Jx.T, -gx, rcond=rcond)
    grad_th = Jth.T @ mu
    return grad_th, mu, Jx, Jth


# ----------------------------------------------------------------------------
# Losses and reference gradients
# ----------------------------------------------------------------------------


def make_linear_loss(N, M, K, seed=7):
    """Random linear functional of ALL outputs; returns (loss_on_x_dict, w)."""
    unflat_x, _ = flatteners(N, M, K)
    d_U = flat(
        dict(
            L=jnp.zeros((N, K)),
            R=jnp.zeros((M, K)),
            alphas=jnp.zeros((K,)),
            betas=jnp.zeros((K - 1,)),
            res=jnp.zeros((M,)),
            c=jnp.zeros(()),
        )
    ).shape[0]
    w = jax.random.normal(jax.random.PRNGKey(seed), (d_U,))

    def loss(xd):
        return jnp.dot(w, flat(xd))

    return loss, w


def autodiff_reference_grad(A, v, K, loss, reortho=True):
    """Ground-truth gradient of loss(forward(A, v)) via plain reverse-mode AD."""
    bd = bidiagonalize(num_matvecs=K, custom_vjp=False, reorthogonalize=reortho)

    def obj(A_, v_):
        out = bd(lambda vec, p: p @ vec, v_, A_)
        xd = dict(L=out.ls, R=out.rs, alphas=out.as_, betas=out.bs, res=out.res, c=jnp.asarray(out.c))
        return loss(xd)

    gA, gv = jax.grad(obj, argnums=(0, 1))(A, v)
    return dict(A=gA, v=gv)


def rel_err(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def random_problem(N, M, K, seed=0):
    kA, kv = jax.random.split(jax.random.PRNGKey(seed))
    A = jax.random.normal(kA, (N, M))
    v = jax.random.normal(kv, (M,))
    return A, v
