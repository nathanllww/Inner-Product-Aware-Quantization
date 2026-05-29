#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "kmeans.h"
#include "interval_sum.hpp"

static double lambda = 0;
static interval_sum<double> is;

static std::vector<double> f;
static std::vector<size_t> bestleft;

kmeans_hirschberg_larmore::kmeans_hirschberg_larmore(const std::vector<double> &points) :
    f(points.size() + 1, 0.0), bestleft(points.size() + 1, 0),
    is(points), points(points), n(points.size()) { }

std::string kmeans_hirschberg_larmore::name() { return std::string("hirc"); }

/**
 * Solves for k clusters using Hirschberg-Larmore with interpolation search
 * on the Lagrange multiplier lambda.  Same outer structure as kmeans_wilber:
 * bracket (lo_k, hi_k) and linearly interpolate to find the next lambda.
 * Delegates to basic() for each DP evaluation.
 */
std::unique_ptr<kmeans_result> kmeans_hirschberg_larmore::compute(size_t k) {
    std::unique_ptr<kmeans_result> kmeans_res(new kmeans_result);
    if (k >= n) {
        kmeans_res->cost = 0.0;
        kmeans_res->centers.resize(k);
        for (size_t i = 0; i < n; ++i) {
            kmeans_res->centers[i] = points[i];
        }
        for (size_t i = n; i < k; ++i) {
            kmeans_res->centers[i] = points[n-1];
        }
        return kmeans_res;
    }
    if (k == 1) {
        kmeans_res->cost = is.cost_interval_l2(0, n-1);
        kmeans_res->centers.push_back(is.query(0, n) / ((double) n));
        return kmeans_res;
    }

    double lo = 0.0;
    double hi = is.cost_interval_l2(0, n-1);
    double hi_intercept = hi;
    double lo_intercept = 0;
    size_t hi_k = 1;
    size_t lo_k = n;
    //double hi = 1e-2;

    double val_found, val_found2;
    size_t k_found, k_found2;
    size_t cnt = 0;
    while (true) {
        ++cnt;
        double t = (hi_intercept - lo_intercept) / sqrt(lo_k - hi_k);
        double intercept_guess = (hi_intercept + lo_intercept) / 2;
        double intersect_hi = (intercept_guess - hi_intercept) / (hi_k - k);
        double intersect_lo = (intercept_guess - lo_intercept) / (lo_k - k);
        assert(intercept_guess > 0);
        assert(intercept_guess <= hi_intercept);
        assert(intercept_guess >= lo_intercept);
        lambda = (hi_intercept - lo_intercept) / (lo_k - hi_k);

        std::tie(val_found, k_found) = this->basic(n);
        if (k_found > k) {
            lo_k = k_found;
            lo = lambda;
            lo_intercept = val_found - lo_k * lambda;
        } else if (k_found < k) {
            hi = lambda;
            hi_k = k_found;
            hi_intercept = val_found - hi_k * lambda;
        } else {
            hi = lambda;
            break;
        }
    }
    assert(k == k_found);
    get_actual_cost(n, kmeans_res);
    return kmeans_res;
}

std::unique_ptr<kmeans_result> kmeans_hirschberg_larmore::compute_and_report(size_t k) {
    return compute(k);
}

/**
 * L2 clustering cost for points[i..j-1] plus lambda penalty per cluster.
 * Returns infinity when i >= j (empty/invalid interval).
 */
double kmeans_hirschberg_larmore::weight(size_t i, size_t j) {
    if (i >= j) return std::numeric_limits<double>::max();
    return is.cost_interval_l2(i, j-1) + lambda;
}

/**
 * DP transition matrix entry: f[i] + weight(i, j).
 * Represents the cost of clustering points[0..i-1] optimally, then grouping
 * points[i..j-1] as one final cluster.
 */
double kmeans_hirschberg_larmore::g(size_t i, size_t j) {
    return f[i] + weight(i, j);
}

/**
 * Hirschberg-Larmore "bridge" test: determines whether row j is dominated by
 * rows i and k in the DP matrix and can be removed from the active deque.
 *
 * Row j is dominated (returns true, meaning "remove j") if, for all future
 * columns m in [k..n], either row i or row k achieves a lower g value than
 * row j.  Uses binary search on m to find the crossover point efficiently.
 *
 * @param i  Row to the left of j in the deque.
 * @param j  Row under consideration for removal.
 * @param k  Row to the right of j (i.e. the newly appended row).
 * @param n  Right boundary of columns to consider.
 * @returns  true if j is dominated and should be deleted.
 */
bool kmeans_hirschberg_larmore::bridge(size_t i, size_t j, size_t k, size_t n) {
    if (k == n) {
        return true;
    }
    if (g(i, n) <= g(j, n)) {
        return true;
    }
    size_t lo = k;
    size_t hi = n;
    while (hi - lo >= 2) {
        size_t mid = lo + (hi-lo)/2;
        double gim = g(i, mid);
        double gjm = g(j, mid);
        double gkm = g(k, mid);
        if (gim <= gjm) {
            lo = mid;
            if (gkm <= gjm) return true;
        } else {
            hi = mid;
            if (gjm < gkm) return false;
        }
    }
    bool result = (g(k, hi) <= g(j, hi));
    return result;
}

/**
 * Runs the Hirschberg-Larmore O(n) single-pass DP for the current lambda.
 *
 * Maintains a deque D of "candidate" predecessor rows.  For each column m:
 *   - f[m] = g(D.front(), m): take the best predecessor from the front.
 *   - Advance the front if the next row is better for m+1.
 *   - Append m as a new candidate if it might help for future columns.
 *   - Prune dominated rows from the back using bridge().
 *
 * Unlike Wilber's SMAWK-based approach, this uses the concave SMAWK property
 * via a deque and the bridge test, running in O(n) per lambda evaluation.
 *
 * @returns (f[n], number_of_clusters) after tracing back through bestleft[].
 */
std::pair<double, size_t> kmeans_hirschberg_larmore::basic(size_t n) {
    std::cout << "call basic lambda=" << lambda << std::endl;
    f.resize(n+1, 0);
    for (size_t i = 0; i <= n; ++i) f[i] = 0;
    bestleft.resize(n+1, 0);
    for (size_t i = 0; i <= n; ++i) bestleft[i] = 0;
    std::vector<size_t> D = {0};
    size_t front = 0;
    for (size_t m = 1; m <= n-1; ++m) {
        f[m] = g(D[front], m);
        bestleft[m] = D[front];
        while (front + 1 < D.size() && g(D[front + 1], m+1) <= g(D[front], m+1)) {
            ++front;
        }
        if (g(m, n) < g(D[D.size() - 1], n)) {
            D.push_back(m);
        } else { continue; }

        while (front + 2 < D.size() && bridge(D[D.size() - 3], D[D.size() - 2], m, n))  {
            std::swap(D[D.size() - 1], D[D.size() - 2]);

            D.pop_back();
        }

        if (front + 2 == D.size() && g(D[D.size() - 1], m+1) <= g(D[D.size() - 2], m+1)) {
            ++front;
        }
    }
    assert(front + 1 == D.size());
    f[n] = g(D[front], n);
    bestleft[n] = D[front];

    // find length.
    size_t m = n;
    size_t length = 0;
    while (m > 0) {
        m = bestleft[m];
        ++length;
    }
    return std::make_pair(f[n], length);
}

/**
 * Reconstructs cluster centers and true L2 cost from bestleft[] by tracing
 * back from n to 0.  Fills res->centers in sorted (ascending) order.
 */
double kmeans_hirschberg_larmore::get_actual_cost(size_t n, std::unique_ptr<kmeans_result> &res) {
    double cost = 0.0;
    size_t m = n;

    std::vector<double> centers;
    while (m != 0) {
        size_t prev = bestleft[m];
        cost += is.cost_interval_l2(prev, m-1);
        double avg = is.query(prev, m) / (m - prev);
        centers.push_back(avg);
        m = prev;
    }

    res->centers.resize(centers.size());
    for (size_t i = 0; i < centers.size(); ++i) {
        res->centers[i] = centers[centers.size() - i - 1];
    }
    res->cost = cost;
    return cost;
}

/**
 * O(n²) brute-force Lagrangian DP: for each column m, tries all rows i < m.
 * Used only for correctness testing of the basic() fast path.
 * @returns (f[n-1], number_of_clusters)
 */
std::pair<double, size_t> kmeans_hirschberg_larmore::traditional(size_t n) {
    f.resize(n, 0);
    for (size_t i = 0; i < n; ++i) f[i] = 0;
    bestleft.resize(n, 0);
    for (size_t i = 0; i < n; ++i) bestleft[i] = 0;

    for (size_t m = 1; m < n; ++m) {
        f[m] = g(0, m);
        bestleft[m] = 0;
        for (size_t i = 1; i < m; ++i) {
            if (g(i, m) < f[m]) {
                f[m] = g(i, m);
                bestleft[m] = i;
            }
        }
    }
    size_t m = n-1;
    size_t length = 0;
    while (m > 0) {
        m = bestleft[m];
        ++length;
    }
    return std::make_pair(f[n-1], length);
}
