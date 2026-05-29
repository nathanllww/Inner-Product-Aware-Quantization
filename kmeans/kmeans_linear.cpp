#include "kmeans.h"
#include <cassert>
#include <limits>
/**
 * This code implements the matrix searching algorithm
 * for computing k-means.
 */
static double oo = std::numeric_limits<double>::max();

kmeans_linear::kmeans_linear(const std::vector<double> &points) : kmeans_dp(points) { }

std::string kmeans_linear::name() { return std::string("linear"); }
std::unique_ptr<kmeans_dp> kmeans_linear::get_instance(std::vector<double> &points) {
    return std::unique_ptr<kmeans_dp>(new kmeans_linear(points));
}

/**
 * Runs the O(kn) SMAWK-based k-means DP.
 * Fills the base case (k=1) then iterates, calling fill_row() for each layer.
 * Each fill_row() call uses mincompute() (SMAWK) to find all row minima in O(n).
 */
std::unique_ptr<kmeans_result> kmeans_linear::compute(size_t k) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);
    res->cost = 0.0;

    base_case(k);
    for (size_t c = 2; c <= k; ++c) {
        std::swap(row_prev, row);
        fill_row(c);
    }
    res->cost = row[n-1];
    return res;
}

/**
 * @param i is the number of clusters / row of the matrix.
 * @param m is the last point of the clustering.
 * @param j is the first point of the last cluster.
 * @return C_i[m][j] in the the 1D kmeans paper.
 */
double kmeans_linear::cimj(size_t i, size_t m, size_t j) {
    assert(i > 0);
    if (m < j) {
        return row_prev[m];
    } else {
        if (j == 0) {
            return is.cost_interval_l2(0, m);
        }
        double best_before = row_prev[j-1];
        double last_cluster_cost = is.cost_interval_l2(j, m);
        return last_cluster_cost + best_before;
    }
}

/**
 * SMAWK column-reduction step: eliminates dominated columns from a totally
 * monotone matrix until at most n columns remain (one per row).
 *
 * Uses a doubly-linked list (prev_col/next_col) to delete columns in-place
 * without shifting, maintaining the staircase invariant as it scans through
 * the matrix.  Rows are addressed via row_multiplier * rowk to support the
 * recursive doubling in mincompute().
 *
 * @param row_multiplier  Stride between logical rows (doubles each recursion).
 * @param cols            Input column set (indices into the points array).
 * @param n               Number of rows in the logical submatrix.
 * @param m               Number of columns before reduction.
 * @param cols_output     Output: surviving columns after reduction (size n).
 * @param reduce_i        The current DP layer (number of clusters) being solved.
 */
void kmeans_linear::reduce(size_t row_multiplier, std::vector<size_t> &cols, size_t n, size_t m,
                         std::vector<size_t> &cols_output, size_t reduce_i) {
    // n rows, m columns.
    // output is n rows and n columns.
#ifdef DEBUG_KMEANS
    printf("[reduce] called with n=%d, m=%d, reduce_i=%d\n", n, m, reduce_i);
#endif
    std::vector<size_t> prev_col(m, 0);
    std::vector<size_t> next_col(m, 0);
    for (size_t i = 1; i < m; ++i) {
        prev_col[i] = i-1;
        next_col[i] = i+1;
    }
    next_col[0] = 1;
    prev_col[0] = m;

    size_t remaining_columns = m;
    size_t rowk = 0; // index in rows
    size_t colk = 0; // index in cols.
    while (remaining_columns > n) {
        double val = -cimj(reduce_i, row_multiplier * rowk, cols[colk]);
        //printf("rowk=%ld\n", rowk);
        double next_val = -cimj(reduce_i, row_multiplier * rowk,
                                cols[next_col[colk]]);
        if (val >= next_val && rowk < n-1) {
            rowk += 1;
            colk = next_col[colk];
            assert(colk < m);
        } else if (val >= next_val && rowk == n-1) {
            // delete column next_col[colk].
            // i.e. update the pointers.
            size_t to_delete = next_col[colk];
            //assert(to_delete < m);
            size_t next = next_col[to_delete];
            size_t prev = prev_col[to_delete];
            assert(prev == colk);
            if (next != m) {
                prev_col[next] = prev;
            }
            if (prev != m) {
                next_col[prev] = next;
            }
            next_col[to_delete] = m;
            prev_col[to_delete] = m;
            --remaining_columns;
        } else if (val < next_val) {
            // First adjust pointers. Need to use old pointers later.
            size_t old_colk = colk;
            if (rowk > 0) {
                --rowk;
                colk = prev_col[colk];
            } else {
                colk = next_col[colk];
            }

            // delete column colk, which means update the pointers.
            size_t prev = prev_col[old_colk];
            size_t next = next_col[old_colk];
            if (prev != m) { // meaning the previous exists.
                assert(next_col[prev] == old_colk);
                next_col[prev] = next_col[old_colk];
            }
            if (next != m) { // meaning next exists.
                assert(prev_col[next] == old_colk);
                prev_col[next] = prev_col[old_colk];
            }
            prev_col[old_colk] = m;
            next_col[old_colk] = m;
            --remaining_columns;
        }
    }

    // generate output.
    size_t j = 0;
    for (size_t i = 0; i < m; ++i) {
        if (prev_col[i] != m || next_col[i] != m) {
            cols_output[j] = cols[i];
            ++j;
        }
    }
}

/**
 * SMAWK row-minima computation: for each of the n rows (spaced row_multiplier
 * apart), finds the column in cols[0..m-1] that minimises cimj().
 *
 * Divide-and-conquer:
 *   1. Reduce m columns down to n.
 *   2. Recurse with doubled row stride to solve for even-indexed rows.
 *   3. Fill odd-indexed rows by linear scan between neighbouring even solutions.
 *
 * @param cols_output  Output: for each row i, cols_output[i] is the column
 *                     index achieving the minimum cimj value.
 */
void kmeans_linear::mincompute(size_t row_multiplier, std::vector<size_t> &cols, size_t n, size_t m,
                             size_t reduce_i, std::vector<size_t> &cols_output) {
#ifdef DEBUG_KMEANS
    printf("[mincompute] Called with n=%d, m=%d, reduce_i=%d\n", n, m, reduce_i);
#endif
    if (n == 1) {
        size_t r = 0; // r = rows[0]
        size_t idx = 0;
        double best = cimj(reduce_i, r, cols[0]);
        for (size_t i = 1; i < m; ++i) {
            size_t c = cols[i];
            double val = cimj(reduce_i, r, c);
            if (val < best) {
                best = val;
                idx = i;
            }
        }
        cols_output[0] = cols[idx];
#ifdef DEBUG_KMEANS
        printf("[mincompute] returning\n", n, m, reduce_i);
#endif
        return;
    }

    std::vector<size_t> cols_output_reduce(n, 0);
    reduce(row_multiplier, cols, n, m, cols_output_reduce, reduce_i);
    size_t n_rec = (n + 1) / 2;
    std::vector<size_t> output(n_rec, 0);
    mincompute(row_multiplier * 2, cols_output_reduce, n_rec, n, reduce_i, output);
#ifdef DEBUG_KMEANS
    printf("[mincompute] n = %d\n", n);
    for (size_t i = 0; i < n_rec; ++i) {
        printf("[mincompute] output[%ld] = %ld\n", i, output[i]);
    }
#endif
    { std::vector<size_t> empty; std::swap(empty, cols_output_reduce); }

    size_t first = 0; // index into cols.
    while (output[0] != cols[first]) ++first;

    // iterate odd rows.
    for (size_t i = 1; i < n; i+=2) {
        size_t current = first; // index into cols
        size_t end = current; // index into cols
        if (i + 1 < n) {
            while (output[(i+1)/2] != cols[end]) ++end;
        } else {
            end = m-1;
        }

        size_t best_idx = current; // index into cols.
        double best = oo;
        for (size_t z = current; z <= end; ++z) {
            size_t rowsi = row_multiplier * i;
            double val = cimj(reduce_i, rowsi, cols[z]);
            if (val < best) {
                best = val;
                best_idx = z;
            }
        }
#ifdef DEBUG_KMEANS
        printf("[mincompute] %d: cols[best_idx] = %d\n", i, cols[best_idx]);
#endif
        cols_output[i] = cols[best_idx];
        first = end;
    }
    for (size_t i = 0; i < n; i+=2) cols_output[i] = output[i/2];
    { std::vector<size_t> empty; std::swap(empty, output); };
#ifdef DEBUG_KMEANS
    printf("[mincompute] reduce_i = %d   n = %d\n", reduce_i, n);
    for (size_t i = 0; i < n; ++i) {
        printf("[mincompute] cols_output[%d] = %d\n", i, cols_output[i]);
    }
    printf("[mincompute] returning\n");
#endif
    return;
}

/**
 * Fills DP row k for all n points using SMAWK (mincompute), running in O(n).
 * After mincompute() returns the optimal split column for each row, evaluates
 * cimj() at each (row, best-column) pair to store the minimum cost.
 */
void kmeans_linear::fill_row(size_t k) {
    std::vector<size_t> cols(n, 0);
    std::vector<size_t> output(n, 0);
    for (size_t i = 0; i < n; ++i) {
        cols[i] = i;
    }

#ifdef DEBUG_KMEANS
    printf("Matrix cimj:\n");
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%.2f  ", cimj(k, i, j));
        }
        printf("\n");
    }
    printf(" ------    fill_row2   k = %ld   ---------\n", k);
#endif
    mincompute(1, cols, n, n, k, output);

    for (size_t i = 0; i < n; ++i) {
#ifdef DEBUG_KMEANS
        printf("output[%d] = %d\n", i, output[i]);
#endif
        size_t row = i;
        size_t col = output[i];
        this->row[i] = cimj(k, row, col);
    }
}

/**
 * Initialises DP row 1: row[i] = L2 cost of clustering points[0..i] into a
 * single cluster.  row[0] = 0 (empty cluster convention used by cimj()).
 */
void kmeans_linear::base_case(size_t k) {
    for (size_t i = 0; i < n; ++i) {
        double cost = is.cost_interval_l2(0, i);
        row[i] = cost;
    }
    row[0] = 0.0;
}

