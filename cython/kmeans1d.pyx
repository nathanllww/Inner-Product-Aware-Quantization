# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

cnp.import_array()

# ---------------------------------------------------------------------------
# C++ declarations
# ---------------------------------------------------------------------------

cdef extern from "kmeans/kmeans.h":
    cdef cppclass kmeans_result:
        double cost
        vector[double] centers
        vector[size_t] path

    cdef cppclass kmeans_wilber:
        kmeans_wilber(const vector[double]& points)
        unique_ptr[kmeans_result] compute(size_t k, double lower_bound, double upper_bound)

    # kmeans_wilber_t<W> is available as a template for custom weight
    # functions defined in C++.  To expose a new weight function to Python:
    #   1. Define a WeightFn struct in a .cpp file (see kmeans_wilber_impl.hpp
    #      for the concept: operator(), total_cost(), center()).
    #   2. Declare the concrete instantiation in a header.
    #   3. Add a cdef extern block here and a Python-callable wrapper below,
    #      following the kmeans_wilber_1d pattern.


# ---------------------------------------------------------------------------
# Python-callable wrapper — L2 k-means
# ---------------------------------------------------------------------------

def kmeans_wilber_1d(
    cnp.ndarray[cnp.float64_t, ndim=1] points,
    Py_ssize_t k,
):
    """Compute optimal 1D k-means using Wilber's O(n log² n) algorithm.

    Parameters
    ----------
    points : ndarray[float64], 1-D, **must be sorted in ascending order**
    k      : number of clusters (1 <= k <= len(points))

    Returns
    -------
    centers : ndarray[float64] of shape (k,)
        Optimal cluster centers, sorted ascending.
    cost : float
        Total sum-of-squared-distances cost.

    Notes
    -----
    The algorithm uses Lagrangian relaxation with SMAWK (totally-monotone
    matrix searching) to solve 1-D k-means exactly.  Input must be sorted;
    no sorting is performed here.
    """
    cdef Py_ssize_t n = points.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError("k cannot exceed the number of points")

    # Ensure C-contiguous float64 array.
    cdef cnp.ndarray[cnp.float64_t, ndim=1] pts = np.ascontiguousarray(points, dtype=np.float64)

    # Copy into a std::vector for the C++ constructor.
    cdef vector[double] pts_vec
    pts_vec.resize(n)
    cdef Py_ssize_t i
    for i in range(n):
        pts_vec[i] = pts[i]

    # Run Wilber's algorithm.
    cdef kmeans_wilber* solver = new kmeans_wilber(pts_vec)
    cdef unique_ptr[kmeans_result] result = solver.compute(<size_t>k, -1.0, -1.0)
    del solver

    # Extract results before the unique_ptr destructs.
    cdef kmeans_result* res_ptr = result.get()
    cdef double cost = res_ptr.cost
    cdef vector[double]* cv = &res_ptr.centers
    cdef Py_ssize_t ncenters = cv.size()

    cdef cnp.ndarray[cnp.float64_t, ndim=1] centers = np.empty(ncenters, dtype=np.float64)
    cdef double[::1] centers_view = centers
    for i in range(ncenters):
        centers_view[i] = cv[0][i]

    return centers, cost
