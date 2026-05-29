#include "kmeans.h"

// kmeans_wilber is a thin subclass of kmeans_wilber_t<L2WeightFn>.
// All algorithm logic (SMAWK, Wilber DP, lambda search, noise-perturbation
// fallback) lives in the template base in kmeans_wilber_impl.hpp.

kmeans_wilber::kmeans_wilber(const std::vector<double> &points)
    : kmeans_wilber_t<L2WeightFn>(points.size(), L2WeightFn(points)) {}

std::string kmeans_wilber::name() { return "wilber"; }
