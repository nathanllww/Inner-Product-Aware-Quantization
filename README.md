Code for the paper "Inner Product Aware Quantization: Provably Fast, Accurate, and Adaptive Algorithms" by N. White and K. Singal

The main code is in the `cython` folder, with the (root directory) file `setup_cython.py` as the build file.
To build, run
```
python setup_cython.py build_ext --inplace
```
Necessary Python dependecies are `cython`, `numpy`, and `setuptools` (for building).

Our code uses and modifies the [kmeans1d](https://github.com/gronlund/kmeans1d) library, which is an implementation of the paper [Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D](https://arxiv.org/abs/1701.07204); this code and our modifications are in the `kmeans` folder.
