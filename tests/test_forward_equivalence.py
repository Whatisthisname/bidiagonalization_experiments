import jax
import jax.numpy as jnp
from matfree import decomp
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algo_hessenberg import arnoldi, hessenberg


def test_forward_equivalence():
    n = 4
    m = 4

    matvecs = 3

    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(n, m))

    A_aug = jnp.block(
        [
            [jnp.zeros((n, n)), A],
            [A.T, jnp.zeros((m, m))],
        ]
    )

    def matvec(v, params):
        A = params
        return A @ v

    v = jax.random.normal(jax.random.PRNGKey(0), shape=(m))
    v_aug = jnp.concat([jnp.zeros(n), v])

    # arnoldi_func = arnoldi(matvecs_num=4, custom_vjp=False, reorthogonalize=True)
    arnoldi_func = hessenberg(2 * matvecs, custom_vjp=True, reortho="full")
    Q, H, res, c = arnoldi_func(matvec, (n, m), v_aug, A_aug)

    print("Q")
    print(Q.round(3))
    print("H")
    print(H.round(3))
    print("res")
    print(res.round(3))
    print("c")
    print(c.round(3))

    print()
    print("Bidiagonalization:")
    bd = decomp.bidiag(matvecs)
    result: decomp._DecompResult = bd(matvec, v, A)
    print("Q_upper")
    print(result.Q_tall[0].round(3))
    print("Q_lower")
    print(result.Q_tall[1].round(3))
    print("B")
    print(result.J_small.round(3))
    print("res")
    print(result.residual.round(3))
    print("c")
    print(result.init_length_inv.round(3))

    rlrlrl = Q
    rs = rlrlrl[n:, 0::2]
    ls = rlrlrl[:n, 1::2]
    ababab = jnp.diag(H, k=1)
    alphas = ababab[::2]
    betas = ababab[1::2]
    res = res[n:]
    stuff1 = (rs, ls, alphas, betas, res, c)
    print(jax.tree.map(jnp.shape, stuff1))

    ls_rs, B, res, c = result
    ls, rs = ls_rs
    alphas = jnp.diag(B)
    betas = jnp.diag(B, k=1)
    stuff2 = (rs, ls, alphas, betas, res, c)
    print(jax.tree.map(jnp.shape, stuff2))

    print(stuff1)
    print("other")
    print(stuff2)

    names = ("rs", "ls", "alphas", "betas", "res", "c")

    def compare(leaf1, leaf2, name):
        assert jnp.allclose(leaf1, leaf2, atol=1e-15), (
            f"{name} differ: error magnitude: {jnp.linalg.norm(leaf1 - leaf2)}"
        )

    jax.tree.map(compare, stuff1, stuff2, names)

    def measure(leaf1, leaf2, name):
        return jnp.linalg.norm(leaf1 - leaf2)

    print(jax.tree.map(measure, stuff1, stuff2, names))
