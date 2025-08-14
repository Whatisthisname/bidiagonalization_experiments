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
    data = jnp.asarray(matrix.data)
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
