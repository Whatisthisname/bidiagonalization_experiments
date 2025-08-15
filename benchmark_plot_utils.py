import matplotlib.pyplot as plt
import scipy
import jax.numpy as jnp
import jax
import os

import jax.experimental.sparse
import jax.flatten_util
import scipy.io


def suite_sparse_download(
    path,
    name_or_id,
    *,
    limit=5,
    isspd=None,
    nzbounds=None,
    rowbounds=(None, None),
    colbounds=(None, None),
    matrixformat="MM",
):
    """Download from https://sparse.tamu.edu/."""
    import ssgetpy

    searched = ssgetpy.search(
        name_or_id=name_or_id,
        limit=limit,
        isspd=isspd,
        nzbounds=nzbounds,
        rowbounds=rowbounds,
        colbounds=colbounds,
    )
    searched.download(destpath=path, format=matrixformat, extract=True)


def suite_sparse_load(which, /, path="./data/matrices/", suffix=".mtx"):
    matrix = scipy.io.mmread(f"{path}{which}/{which}{suffix}")

    row = jnp.asarray(matrix.row)
    col = jnp.asarray(matrix.col)
    data = jnp.asarray(matrix.data).astype(float)
    indices = jnp.stack([row, col]).T
    return jax.experimental.sparse.BCOO([data, indices], shape=matrix.shape)


def plt_spy_coo(
    ax, A, /, markersize=3, cmap="jet", invert_axes=True, subsample: int = 1
):
    """Plot the sparsity pattern of a BCOO matrix.

    Credit:
    https://gist.github.com/lukeolson/9710288
    """
    ax.scatter(
        A.indices[::subsample, 0],
        A.indices[::subsample, 1],
        c=A.data[::subsample],
        s=markersize,
        marker="s",
        edgecolors="none",
        clip_on=False,
        cmap=cmap,
    )
    nrows, ncols = A.shape
    ax.set_xlim((0, nrows))
    ax.set_ylim((0, ncols))

    if invert_axes:
        ax.invert_yaxis()
        ax.xaxis.tick_top()


if __name__ == "__main__":
    ### Example of using sparse matrix:

    which_matrix = "gyro"

    path = "./data/matrices/"
    suite_sparse_download(path=path, name_or_id=which_matrix)
    M = suite_sparse_load(which_matrix, path=path)

    params, params_unflatten = jax.flatten_util.ravel_pytree(M.data)

    @jax.jit
    def matvec(v, p):
        pp = params_unflatten(p)
        matrix = jax.experimental.sparse.BCOO((pp, M.indices), shape=M.shape)
        return matrix @ v

    v = jnp.ones(M.shape[0])

    jax.vjp(matvec, v, params)[1](matvec(v, params))
