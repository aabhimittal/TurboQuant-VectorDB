"""Input validation shared by every quantizer and index.

Why a whole module for this: in a vector database, bad input does not
crash -- it silently returns wrong neighbors. A single NaN in a training
batch propagates through `np.percentile` into every range, through
k-means into every centroid, and the index then answers every query with
garbage while looking perfectly healthy. An id typo'd into the wrong
dtype makes a fancy-index gather return the wrong row, not an error.

So the rule here is: reject at the boundary, loudly, with a message that
names the offending row. Every public `train`/`add`/`search`/`encode`
entry point funnels its array through `as_2d_float32` first.

The checks are cheap relative to what follows them (a `np.isfinite`
reduction is one pass over data that k-means is about to touch 25 times),
and they are the difference between a stack trace at ingest and a silent
recall regression discovered a month later.
"""

from __future__ import annotations

import numpy as np


def as_2d_float32(
    data: object,
    *,
    name: str = "data",
    dim: int | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    """Coerce input to a contiguous (n, d) float32 array, or raise.

    Accepts anything array-like (lists, float64 arrays, F-ordered slices,
    non-contiguous views) and normalizes it, because callers legitimately
    hold data in all of those forms. Rejects what cannot be silently
    fixed: wrong rank, wrong width, non-finite values, and -- unless
    `allow_empty` -- zero rows.

    Args:
        data: array-like to validate.
        name: label used in error messages (e.g. "queries", "vectors").
        dim: expected width; checked when not None.
        allow_empty: permit n == 0 (used by `add`, which may legally be
            handed an empty batch by a streaming pipeline).

    Returns:
        A C-contiguous float32 array of shape (n, d).

    Raises:
        TypeError: input is not numeric.
        ValueError: wrong rank, wrong width, empty when disallowed, or
            containing NaN/Inf.
    """
    try:
        arr = np.asarray(data)
    except Exception as exc:  # pragma: no cover - exotic array-likes
        raise TypeError(f"{name} could not be interpreted as an array: {exc}") from exc

    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            f"{name} must be numeric, got dtype {arr.dtype!r}. "
            "Ragged/nested lists produce object arrays -- check row lengths."
        )

    # 1-D input is the single most common call-site slip (`index.search(q, k)`
    # with one query). Promote it rather than failing: the intent is
    # unambiguous, unlike a 3-D array.
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")

    # Narrowing float64 -> float32 can overflow (1e308 becomes inf). That is
    # expected here and is caught by the finiteness check below, so silence
    # the RuntimeWarning rather than letting a handled case spam the logs.
    with np.errstate(over="ignore", invalid="ignore"):
        arr = np.ascontiguousarray(arr, dtype=np.float32)

    if arr.shape[0] == 0 and not allow_empty:
        raise ValueError(f"{name} is empty (0 rows)")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(
            f"{name} has dimension {arr.shape[1]}, expected {dim}"
        )

    # float64 -> float32 narrowing can itself manufacture infinities
    # (1e308 overflows), so check finiteness *after* the cast, not before.
    if arr.size and not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        row, col = int(bad[0, 0]), int(bad[0, 1])
        raise ValueError(
            f"{name} contains non-finite values (first at row {row}, column "
            f"{col}: {arr[row, col]}). NaN/Inf silently corrupt centroids and "
            "quantization ranges -- clean or drop these rows before indexing."
        )
    return arr


def check_k(k: object, *, name: str = "k") -> int:
    """Validate a neighbor count: a positive Python int.

    `k=0` returns nothing, `k=-1` silently reverses slicing semantics, and
    `k=2.5` breaks `np.argpartition` deep in the call stack. All three are
    caller bugs worth naming at the surface.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(k).__name__}")
    k = int(k)
    if k < 1:
        raise ValueError(f"{name} must be >= 1, got {k}")
    return k


def check_ids(ids: object, ntotal: int, *, name: str = "ids") -> np.ndarray:
    """Validate an id array against the index's current size.

    Out-of-range ids are the dangerous case: NumPy fancy-indexing accepts
    negatives (wrapping to the end of the array) and only raises above the
    upper bound, so `remove_ids([-1])` would quietly delete the newest
    vector instead of erroring.
    """
    arr = np.asarray(ids)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype, got {arr.dtype!r}")
    arr = arr.astype(np.int64, copy=False)
    if arr.min() < 0 or arr.max() >= ntotal:
        raise IndexError(
            f"{name} out of range: values must lie in [0, {ntotal}), got "
            f"[{int(arr.min())}, {int(arr.max())}]"
        )
    return arr


def check_dim(dim: object) -> int:
    """Validate a dimensionality at construction time."""
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
        raise TypeError(f"dim must be an integer, got {type(dim).__name__}")
    dim = int(dim)
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    return dim


def warn_small_training_set(n: int, k: int, what: str) -> None:
    """Raise when a codebook is being trained on fewer points than centroids.

    k-means cannot place k centroids from n < k points, and even n slightly
    above k produces codebooks that memorize the training sample. FAISS
    uses 39 points/centroid as its warning threshold; we hard-fail only on
    the impossible case and leave the merely-unwise case to the caller.
    """
    if n < k:
        raise ValueError(
            f"{what} needs at least {k} training points (one per centroid), "
            f"got {n}. Reduce the codebook size or supply more training data."
        )
