# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

cnp.import_array()

# ---------------------------------------------------------------------------
# kmeans_result (declared here so Cython knows its fields)
# ---------------------------------------------------------------------------

cdef extern from "kmeans/kmeans.h":
    cdef cppclass kmeans_result:
        double cost
        vector[size_t] path   # split indices [n, ..., 0]; reversed = boundaries

    cdef cppclass kmeans:
        unique_ptr[kmeans_result] compute(size_t k, double lower_bound, double upper_bound)
        unique_ptr[kmeans_result] compute_with_binary_search(size_t k, double lower_bound, double upper_bound)

cdef extern from "asq_weight_fns.h":
    cdef cppclass ASQWeightFn:
        ASQWeightFn(const vector[double]& pts, const vector[int]& pt_weights)

    cdef cppclass kmeans_asq_t:
        kmeans_asq_t(size_t n, ASQWeightFn wfn)
        unique_ptr[kmeans_result] compute(size_t k, double lower_bound, double upper_bound)
        unique_ptr[kmeans_result] compute_with_binary_search(size_t k, double lower_bound, double upper_bound)

    cdef cppclass ASQSubsetWeightFn:
        ASQSubsetWeightFn(const vector[double]& w, const vector[double]& x)

    cdef cppclass kmeans_asq_subset_t:
        kmeans_asq_subset_t(size_t n, ASQSubsetWeightFn wfn)
        unique_ptr[kmeans_result] compute(size_t k, double lower_bound, double upper_bound)
        unique_ptr[kmeans_result] compute_with_binary_search(size_t k, double lower_bound, double upper_bound)


# ---------------------------------------------------------------------------
# Shared helper: run solver and extract result
# ---------------------------------------------------------------------------

cdef object _run_and_extract(kmeans* solver, size_t k, double lower_bound,
                              double upper_bound, bint binary_search):
    """Run a kmeans solver and return (splits, cost).  Takes ownership of
    *solver* (deletes it before returning)."""
    cdef unique_ptr[kmeans_result] result
    if not binary_search:
        result = solver.compute(k, lower_bound, upper_bound)
    else:
        result = solver.compute_with_binary_search(k, lower_bound, upper_bound)
    del solver

    cdef kmeans_result* res_ptr = result.get()
    cdef double cost = res_ptr.cost
    cdef vector[size_t]* pv = &res_ptr.path
    cdef Py_ssize_t npath = pv.size()
    cdef Py_ssize_t i
    cdef cnp.ndarray[cnp.int64_t, ndim=1] splits = np.empty(npath, dtype=np.int64)
    for i in range(npath):
        splits[i] = <cnp.int64_t>pv[0][npath - 1 - i]
    return splits, cost


# ---------------------------------------------------------------------------
# Python-callable wrappers
# ---------------------------------------------------------------------------

def kmeans_asq_1d(
    cnp.ndarray[cnp.float64_t, ndim=1] points,
    cnp.ndarray[cnp.int32_t, ndim=1] pt_weights,
    Py_ssize_t k,
    binary_search = False,
    double lower_bound = -1.0,
    double upper_bound = -1.0
):
    """Partition a sorted array into k contiguous clusters minimising total
    cluster-sum cost, solved exactly in O(n log² n) via Wilber's algorithm.

    Parameters
    ----------
    points     : ndarray[float64], 1-D, **must be sorted ascending**
    pt_weights : ndarray[int32], 1-D, per-point weights
    k          : number of clusters (1 <= k <= len(points))

    Returns
    -------
    splits : ndarray[int64], shape (k+1,)
        Ascending boundary indices of the optimal partition.
    cost : float
    """
    cdef Py_ssize_t n = points.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError("k cannot exceed the number of points")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] pts = np.ascontiguousarray(points, dtype=np.float64)

    cdef vector[double] pts_vec
    cdef vector[int] pt_weights_vec
    pts_vec.resize(n)
    pt_weights_vec.resize(n)
    cdef Py_ssize_t i
    for i in range(n):
        pts_vec[i] = pts[i]
        pt_weights_vec[i] = pt_weights[i]

    cdef ASQWeightFn* wfn = new ASQWeightFn(pts_vec, pt_weights_vec)
    cdef kmeans_asq_t* solver = new kmeans_asq_t(<size_t>n, wfn[0])
    del wfn
    return _run_and_extract(<kmeans*>solver, <size_t>k, lower_bound, upper_bound, binary_search)


def kmeans_asq_1d_subset(
    cnp.ndarray[cnp.float64_t, ndim=1] w,
    cnp.ndarray[cnp.float64_t, ndim=1] x,
    Py_ssize_t k,
    binary_search = False,
    double lower_bound = -1.0,
    double upper_bound = -1.0
):
    """Partition a uniform grid x into k contiguous clusters minimising total
    ASQ cost, where cluster costs are computed from the (unsorted) data w.

    Parameters
    ----------
    w : ndarray[float64], 1-D, unsorted data points
    x : ndarray[float64], 1-D, sorted uniform grid; x[i] = (max(w)-min(w))*i/L
    k : number of clusters (1 <= k <= len(x))

    Returns
    -------
    splits : ndarray[int64], shape (k+1,)
        Ascending boundary indices into x of the optimal partition.
    cost : float
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t d = w.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError("k cannot exceed the length of the grid")

    cdef vector[double] w_vec
    cdef vector[double] x_vec
    w_vec.resize(d)
    x_vec.resize(n)
    cdef Py_ssize_t i
    for i in range(d):
        w_vec[i] = w[i]
    for i in range(n):
        x_vec[i] = x[i]

    cdef ASQSubsetWeightFn* wfn = new ASQSubsetWeightFn(w_vec, x_vec)
    cdef kmeans_asq_subset_t* solver = new kmeans_asq_subset_t(<size_t>n, wfn[0])
    del wfn
    return _run_and_extract(<kmeans*>solver, <size_t>k, lower_bound, upper_bound, binary_search)
