"""Child process: jax autodiff through the reorthogonalized forward loop at
the precision fixed by JAX_ENABLE_X64. Reads argv[1] (.npz), writes argv[2]."""

import sys

import numpy as np
import jax
import jax.numpy as jnp

import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo_bidiag import bidiagonalize  # noqa: E402

x64 = bool(jax.config.read("jax_enable_x64"))
ftype = np.float64 if x64 else np.float32

data = np.load(sys.argv[1])
A = jnp.asarray(data["A"].astype(ftype))
v = jnp.asarray(data["v"].astype(ftype))
K = int(data["K"])
g = {k: jnp.asarray(data[k].astype(ftype)) for k in ["gL", "gR", "ga", "gb", "gres"]}
gc = jnp.asarray(data["gc"].astype(ftype))

bd = bidiagonalize(num_matvecs=K, custom_vjp=False, reorthogonalize=True)


def loss(A_, v_):
    out = bd(lambda vec, p: p @ vec, v_, A_)
    s = jnp.sum(out.ls * g["gL"]) + jnp.sum(out.rs * g["gR"])
    s += jnp.sum(out.as_ * g["ga"]) + jnp.sum(out.bs * g["gb"])
    s += jnp.sum(out.res * g["gres"]) + out.c * gc
    return s


gA, gv = jax.grad(loss, argnums=(0, 1))(A, v)
np.savez(sys.argv[2], gA=np.asarray(gA), gv=np.asarray(gv))
