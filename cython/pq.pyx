# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.string cimport memcpy
from libc.string cimport memset

from quant_cython_adv import exact_adv, vmix_approx
from quant_cython_mdv import mdv

cnp.import_array()


cdef inline void block_quant_c(
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X,
    Py_ssize_t m,
    int s,
    str quant_method,
    double* result_ptr,
    double eps,
):
    """
    Core C-level block quantization.

    X must be the transpose of the original data matrix: shape (d, n),
    C-contiguous, so that each group of m consecutive rows is a contiguous
    block of n*m doubles in memory.  The wrapper transposes once so that
    each block can be extracted with a single memcpy instead of n strided ones.

    result_ptr must point to at least ceil(d/m)*s pre-allocated doubles.
    """
    cdef Py_ssize_t d = X.shape[0]
    cdef Py_ssize_t n = X.shape[1]
    cdef Py_ssize_t b = 0, row_start = 0, row_end, block_rows, nm, i, q_len
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] block_data, w_sorted, Q_arr
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] weights
    cdef double* q_ptr

    while row_start < d:
        row_end = row_start + m
        if row_end > d:
            row_end = d
        block_rows = row_end - row_start
        nm = n * block_rows

        # X is d x n (transposed), so rows row_start..row_end-1 are a single
        # contiguous run of nm doubles — one memcpy grabs the entire block.
        block_data = np.empty(nm, dtype=np.float64)
        memcpy(&block_data[0], &X[row_start, 0], nm * sizeof(double))

        if quant_method == 'adv':
            w_sorted = np.sort(block_data)
            weights = np.ones(nm, dtype=np.int32)
            Q_arr = vmix_approx(w_sorted, s, 10*s, weights=weights)[0]
            memcpy(&result_ptr[b * s], &Q_arr[0], s * sizeof(double))
        else:  # mdv
            Q_arr = np.ascontiguousarray(mdv(block_data, s, eps), dtype=np.float64)
            q_len = Q_arr.shape[0]
            q_ptr = &Q_arr[0]
            memcpy(&result_ptr[b * s], q_ptr, q_len * sizeof(double))
            # mdv returns at most s points; pad remainder with the last value.
            for i in range(q_len, s):
                result_ptr[b * s + i] = q_ptr[q_len - 1]

        b += 1
        row_start = row_end


@cython.boundscheck(False)
@cython.wraparound(False)
def block_quant(X, Py_ssize_t m, int s, str quant_method='adv', double eps=0.01):
    """
    Block quantization on an n x d data matrix.

    For each block of m consecutive columns in X, concatenates all n*m values
    into a 1D vector and computes a quantization set of size s using the
    specified method.  The last block may have fewer than m columns when d is
    not divisible by m.

    Parameters
    ----------
    X : array-like, shape (n, d), float64
        Input data matrix.
    m : int
        Block size (number of columns per block).
    s : int
        Quantization set size per block. Must be >= 2.
    quant_method : str, default 'adv'
        'adv' — exact_adv (ASQ / adaptive dispersion variance, exact).
        'mdv' — mdv (max-variance objective, (1+eps)-approximate).
    eps : float, default 0.01
        Approximation factor passed to mdv when quant_method='mdv'.

    Returns
    -------
    ndarray, shape (n_blocks, s), float64
        Row b holds the s quantization points for block b.
        n_blocks = ceil(d / m).
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64).T)
    cdef Py_ssize_t d = X_arr.shape[0]
    cdef Py_ssize_t n = X_arr.shape[1]

    if m <= 0:
        raise ValueError("m must be positive")
    if s < 2:
        raise ValueError("s must be at least 2")
    if quant_method not in ('adv', 'mdv'):
        raise ValueError("quant_method must be 'adv' or 'mdv'")
    if n < 1 or d < 1:
        raise ValueError("X must be non-empty")

    cdef Py_ssize_t n_blocks = (d + m - 1) // m
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] result = np.empty((n_blocks, s), dtype=np.float64)

    # X_arr is d x n (transposed); block_quant_c expects this layout so each
    # block of m dimensions is a contiguous run extractable with one memcpy.
    block_quant_c(X_arr, m, s, quant_method, &result[0, 0], eps)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def batch_row_quant(X, int s, int m, str quant_method='adv', double eps=0.01,
                    weights_in=None):
    """
    Quantize each row of X independently, returning one quantization set per row.

    For quant_method='adv' with no weights_in, all rows are sorted in a single
    vectorized NumPy call and a shared uniform weights array is reused per row.
    When weights_in is provided, argsort is used so that per-row weights can be
    reordered to match each row's sorted order before passing to vmix_approx.

    Parameters
    ----------
    X : array-like, shape (N, D), float64
        Input data. Each row is treated as an independent 1-D dataset.
    s : int
        Number of quantization points per row. Must be >= 2.
    m : int
        Number of vmix_approx candidates (quant_method='adv' only).
    quant_method : str, default 'adv'
        'adv' — vmix_approx. 'mdv' — mdv.
    eps : float, default 0.01
        Approximation factor for mdv (quant_method='mdv').
    weights_in : array-like, shape (D,), float64, optional
        Per-coordinate importance weights (quant_method='adv' only).
        Element j is the weight for coordinate j of each row. For each row the
        weights are reordered to match the row's sorted order before being
        passed to vmix_approx. If None, all coordinates are weighted equally.

    Returns
    -------
    ndarray, shape (N, s), float64
        result[i] is the sorted quantization set for row i.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X_arr = np.ascontiguousarray(
        np.asarray(X, dtype=np.float64)
    )
    cdef Py_ssize_t N = X_arr.shape[0]
    cdef Py_ssize_t D = X_arr.shape[1]

    if s < 2:
        raise ValueError("s must be at least 2")
    if quant_method not in ('adv', 'mdv'):
        raise ValueError("quant_method must be 'adv' or 'mdv'")

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] result = np.empty((N, s), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] weights
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_weights
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X_sorted
    cdef cnp.ndarray[cnp.int64_t,   ndim=2, mode='c'] X_argsorted
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] Q_arr
    cdef Py_ssize_t i, q_len

    if quant_method == 'adv':
        if weights_in is not None:
            weights = np.ascontiguousarray(weights_in, dtype=np.float64)
            X_argsorted = np.argsort(X_arr, axis=1).astype(np.int64, copy=False)
            X_sorted = np.take_along_axis(X_arr, X_argsorted, axis=1)
            for i in range(N):
                row_weights = np.ascontiguousarray(weights[X_argsorted[i]])
                Q_arr = vmix_approx(X_sorted[i], s, m, weights=row_weights)[0]
                result[i] = Q_arr
        else:
            X_sorted = np.sort(X_arr, axis=1)
            weights = np.ones(D, dtype=np.float64)
            for i in range(N):
                Q_arr = vmix_approx(X_sorted[i], s, m, weights=weights)[0]
                result[i] = Q_arr
    else:  # mdv
        for i in range(N):
            Q_arr = np.asarray(mdv(X_arr[i], s, eps), dtype=np.float64)
            q_len = Q_arr.shape[0]
            result[i, :q_len] = Q_arr
            if q_len < s:
                result[i, q_len:] = Q_arr[q_len - 1]

    return result


def _quantize_row_worker(shm_name, shape, dtype_str, start, end, s, m, quant_method, eps):
    """
    Multiprocessing worker that quantizes X[start:end] via shared memory.

    Defined in quant_cython_pq (an importable module) so that both fork-based
    (Linux) and spawn-based (macOS) workers can pickle and resolve it by name,
    avoiding the hyphen-in-filename issue that breaks __main__-scoped functions
    under spawn.
    """
    from multiprocessing.shared_memory import SharedMemory
    shm = SharedMemory(name=shm_name, create=False)
    try:
        X_view = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        # Copy slice before closing so the buffer stays valid.
        X_chunk = np.array(X_view[start:end], dtype=np.float64)
    finally:
        shm.close()
    return batch_row_quant(X_chunk, s, m, quant_method=quant_method, eps=eps)


@cython.boundscheck(False)
@cython.wraparound(False)
def pq_search_topk(tables, packed_T, int k):
    """
    ADC search for block_quant PQ using precomputed lookup tables (MIPS).

    Parameters
    ----------
    tables : array-like, shape (n_queries, n_groups, 256), float32
        Per-query lookup tables where tables[q, g, code] is the inner product
        contribution of group g for packed code value ``code``.
    packed_T : array-like, shape (n_groups, n_db), uint8
        Transposed packed database codes.
    k : int
        Number of top inner product results to return per query.

    Returns
    -------
    ndarray, shape (n_queries, k), int64
        Unsorted top-k indices per query (highest inner products).
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode='c'] tables_arr = np.ascontiguousarray(
        np.asarray(tables, dtype=np.float32)
    )
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode='c'] packed_arr = np.ascontiguousarray(
        np.asarray(packed_T, dtype=np.uint8)
    )
    cdef Py_ssize_t n_queries = tables_arr.shape[0]
    cdef Py_ssize_t n_groups = tables_arr.shape[1]
    cdef Py_ssize_t n_codes = tables_arr.shape[2]
    cdef Py_ssize_t n_db = packed_arr.shape[1]
    cdef Py_ssize_t q, g, i
    cdef float* dptr
    cdef cnp.float32_t* table_ptr
    cdef cnp.uint8_t* code_ptr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode='c'] dists
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] out
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode='c'] topk_idx

    if n_codes != 256:
        raise ValueError("tables must have shape (n_queries, n_groups, 256)")
    if packed_arr.shape[0] != n_groups:
        raise ValueError("packed_T first dimension must match tables n_groups")
    if k <= 0 or k > n_db:
        raise ValueError("k must satisfy 1 <= k <= packed_T.shape[1]")

    dists = np.empty(n_db, dtype=np.float32)
    out = np.empty((n_queries, k), dtype=np.int64)
    dptr = &dists[0]

    for q in range(n_queries):
        memset(dptr, 0, n_db * sizeof(cnp.float32_t))
        for g in range(n_groups):
            table_ptr = &tables_arr[q, g, 0]
            code_ptr = &packed_arr[g, 0]
            for i in range(n_db):
                dptr[i] += table_ptr[code_ptr[i]]

        topk_idx = np.argpartition(-dists, k)[:k]
        out[q, :] = topk_idx

    return out
