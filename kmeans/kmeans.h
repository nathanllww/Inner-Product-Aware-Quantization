#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <stdlib.h>
#include <stdio.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>

#include "interval_sum.hpp"

/**
 * Result of a k-means computation.
 *
 * cost    - total sum-of-squared-distances (L2) clustering cost
 * centers - sorted cluster centers, one per cluster
 * path    - internal back-pointer array from Wilber's DP (bestleft[]);
 *           not meaningful outside of the algorithm itself
 */
class kmeans_result {
public:
  double cost;
  std::vector<double> centers;
  std::vector<size_t> path;
};

/**
 * Abstract base class for all k-means solvers.
 * All implementations take a sorted point array in their constructor.
 *
 * compute(k)            - solve for exactly k clusters; returns a kmeans_result
 * compute_and_report(k) - like compute(k) but also prints per-cluster details
 */
class kmeans {
public:
    virtual ~kmeans() {};
    virtual std::unique_ptr<kmeans_result> compute(size_t k, double lower_bound, double upper_bound) = 0;
    virtual std::unique_ptr<kmeans_result> compute_with_binary_search(size_t k, double lower_bound, double upper_bound) = 0;
    virtual std::unique_ptr<kmeans_result> compute_and_report(size_t k, double lower_bound, double upper_bound) = 0;
    virtual std::string name() = 0;
};

/**
 * Which outer search strategy to use when hunting for the Lagrange multiplier
 * lambda that makes Wilber's DP return exactly k clusters.
 *
 * INTERPOLATION (default) - linear interpolation between known bounds;
 *                           converges in ~20-50 iterations in practice.
 * BINARY                  - plain binary search; slower but simpler.
 *                           Used only for benchmarking in the paper.
 */
enum search_strategy {BINARY = 0, INTERPOLATION = 1};

// Pull in the template base (kmeans_wilber_t<W>) and the WeightFn concept.
// This header uses kmeans, kmeans_result, and search_strategy defined above.
#include "kmeans_wilber_impl.hpp"

/**
 * L2 weight functor: cost of assigning a contiguous subarray of sorted
 * doubles to one cluster equals the within-cluster sum of squared deviations
 * from the mean.  Used by the default kmeans_wilber solver.
 *
 * Satisfies the WeightFn concept required by kmeans_wilber_t<W>:
 *   operator()(i, j) -- L2 cost of points[i..j-1]
 *   total_cost()     -- L2 cost of the full array in one cluster
 *   center(i, j)     -- arithmetic mean of points[i..j-1]
 */
struct L2WeightFn {
    interval_sum<double> is;
    std::vector<double>  pts;   // kept so the noise fallback can perturb data

    explicit L2WeightFn(const std::vector<double> &points)
        : is(points), pts(points) {}

    double operator()(size_t i, size_t j) const {
        return is.cost_interval_l2(i, j - 1);
    }
    double total_cost() const {
        return is.cost_interval_l2(0, pts.size() - 1);
    }
    double center(size_t i, size_t j) const {
        return is.query(i, j) / static_cast<double>(j - i);
    }
};

/**
 * Optimal 1-D k-means solver using Wilber's algorithm.
 *
 * Complexity: O(n log² n) time, O(n) space.
 *
 * This is a thin subclass of kmeans_wilber_t<L2WeightFn>.  It adds the
 * noise-perturbation fallback that handles the rare numerical-degeneracy
 * case where the interpolation search gets stuck in an infinite loop.
 *
 * Usage
 * -----
 *   std::vector<double> pts = {1.0, 2.0, 5.0, 8.0, 9.0};  // must be sorted
 *   kmeans_wilber solver(pts);
 *   auto result = solver.compute(2);
 *   // result->centers == {1.5, 7.333...}, result->cost == ...
 *
 * Adding a new weight function
 * ----------------------------
 * Define a struct satisfying the WeightFn concept (see kmeans_wilber_impl.hpp)
 * and instantiate kmeans_wilber_t<YourWeightFn>.  The SMAWK + Wilber DP code
 * is shared; only the three methods of the functor need to be written.
 */
class kmeans_wilber : public kmeans_wilber_t<L2WeightFn> {
public:
    /** Construct solver for the given sorted point array. */
    explicit kmeans_wilber(const std::vector<double> &points);

    std::string name() override;
};

// these methods aren't needed for our code; we're just using wilber

// class kmeans_hirschberg_larmore : public kmeans {
// public:
//     kmeans_hirschberg_larmore(const std::vector<double> &points);
//     std::unique_ptr<kmeans_result> compute(size_t k, double lower_bound, double upper_bound) override;
//     std::unique_ptr<kmeans_result> compute_and_report(size_t k, double lower_bound, double upper_bound) override;
//     std::string name() override;
// private:
//     bool bridge(size_t i, size_t j, size_t k, size_t n);
//     std::pair<double, size_t> basic(size_t n);

//     std::pair<double, size_t> traditional(size_t n);

//     double get_actual_cost(size_t n, std::unique_ptr<kmeans_result> &res);

//     double weight(size_t i, size_t j);
//     double g(size_t i, size_t j);

//     double lambda;
//     std::vector<double> f;
//     std::vector<std::size_t> bestleft;
//     interval_sum<double> is;
//     std::vector<double> points;
//     std::size_t n;
// };


//class kmeans_dp : public kmeans {
//public:
//    kmeans_dp(const std::vector<double> &points);
//    virtual ~kmeans_dp() {};
//    virtual std::unique_ptr<kmeans_result> compute(size_t k, double lower_bound, double upper_bound) = 0;
//    std::unique_ptr<kmeans_result> compute_and_report(size_t k, double lower_bound, double upper_bound) override;
//    virtual std::unique_ptr<kmeans_dp> get_instance(std::vector<double> &points) = 0;
//  double report(std::vector<double> &points, size_t k, std::vector<double> &centers);
//  //protected:
//    interval_sum<double> is;
//    std::vector<double> row;
//    std::vector<double> row_prev;
//    std::vector<double> points;
//    size_t n;

//    //virtual std::unique_ptr<kmeans_dp> get_new_instance(std::vector<double> &points) = 0;
//};

// class kmeans_linear : public kmeans_dp {
// public:
//     kmeans_linear(const std::vector<double> &points);
//     std::unique_ptr<kmeans_result> compute(size_t k, double lower_bound, double upper_bound) override;
//     std::unique_ptr<kmeans_dp> get_instance(std::vector<double> &points) override;
//     std::string name() override;
// private:
//     double cimj(size_t i, size_t m, size_t j);
//     void reduce(size_t row_multiplier, std::vector<size_t> &cols, size_t n, size_t m,
//                 std::vector<size_t> &cols_output, size_t reduce_i);
//     void mincompute(size_t row_multiplier, std::vector<size_t> &cols, size_t n, size_t m,
//                     size_t reduce_i, std::vector<size_t> &cols_output);
//     void fill_row(size_t k);
//     void base_case(size_t k);
// };

// class kmeans_monotone : public kmeans_dp {
// public:
//     kmeans_monotone(const std::vector<double> &points);
//     std::unique_ptr<kmeans_result> compute(size_t k) override;
//     std::unique_ptr<kmeans_dp> get_instance(std::vector<double> &points) override;
//     std::string name() override;
// private:

//     void fill_row_rec(size_t begin, size_t end, size_t k,
//                       int64_t split_left, int64_t split_right);
//     void fill_row(size_t k);
// };

// class kmeans_slow : public kmeans_dp {
// public:
//     kmeans_slow(const std::vector<double> &points);
//     std::unique_ptr<kmeans_result> compute(size_t k) override;
//     std::unique_ptr<kmeans_dp> get_instance(std::vector<double> &points) override;
//     std::string name() override;
// };

// class kmeans_lloyd : public kmeans {
// public:
//     kmeans_lloyd();
//     virtual std::unique_ptr<kmeans_result> compute(size_t k) = 0;
//     virtual std::unique_ptr<kmeans_result> compute_and_report(size_t k) = 0;
//     virtual void set_seed(std::mt19937::result_type val);
//     std::mt19937::result_type random_value();
//     std::vector<size_t> init_splits(size_t n, size_t k);
//     virtual std::string name() override;
// private:
//     std::mt19937_64 mt;
// };

// class kmeans_lloyd_slow : public kmeans_lloyd {
// public:
//     kmeans_lloyd_slow(const std::vector<double> &points);
//     std::unique_ptr<kmeans_result> compute(size_t k) override;
//     std::unique_ptr<kmeans_result> compute_and_report(size_t k) override;
// private:
//     std::vector<double> points;
//     interval_sum<double> is;
//     size_t n;
// };

// class kmeans_lloyd_fast : public kmeans_lloyd {
// public:
//     kmeans_lloyd_fast(const std::vector<double> &points);
//     std::unique_ptr<kmeans_result> compute(size_t k) override;
//     std::unique_ptr<kmeans_result> compute_and_report(size_t k) override;
// private:
//     std::vector<double> points;
//     interval_sum<double> is;
//     size_t n;
// };

// typedef double (*kmeans_fn)(double *points, size_t n,
//                             double *last_row, size_t k);


// double report_clusters(double *points, size_t n,
//                        double *centers, size_t k,
//                        kmeans_fn);

#endif /* __KMEANS_H__ */
