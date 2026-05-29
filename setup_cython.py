from setuptools import Extension, setup
import numpy as np
from Cython.Build import cythonize
import os
import platform
import subprocess
import tempfile


def _openmp_flags():
    """Return (compile_flags, link_flags) for OpenMP, or ([], []) if unavailable."""
    system = platform.system()

    if system == "Darwin":
        brew_paths = ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
        candidates = [
            (
                ["-Xpreprocessor", "-fopenmp", f"-I{p}/include"],
                [f"-L{p}/lib", "-lomp"],
            )
            for p in brew_paths
        ]
        candidates.append((["-fopenmp"], ["-lgomp"]))
    else:
        candidates = [
            (["-fopenmp"], ["-lgomp"]),
            (["-fopenmp"], []),
        ]

    src = b"#include <omp.h>\nint main(){return omp_get_max_threads();}\n"
    cc = os.environ.get("CC", "cc")
    for cflags, lflags in candidates:
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            f.write(src)
            tmp = f.name
        out = tmp[:-2]  # strip ".c"
        r = subprocess.run(
            [cc] + cflags + [tmp, "-o", out] + lflags,
            capture_output=True,
        )
        try:
            os.unlink(tmp)
        except OSError:
            pass
        try:
            os.unlink(out)
        except OSError:
            pass
        if r.returncode == 0:
            return cflags, lflags
    return [], []


omp_cflags, omp_lflags = _openmp_flags()
if omp_cflags:
    print(f"[setup] OpenMP enabled: cflags={omp_cflags} lflags={omp_lflags}")
    k_center_compile_args = ["-O3", "-march=native"] + omp_cflags
    k_center_link_args = omp_lflags
    k_center_macros = [("HAVE_OPENMP", "1")]
else:
    print("[setup] WARNING: OpenMP not found. parallized functions will be single-threaded.")
    k_center_compile_args = ["-O3", "-march=native"]
    k_center_link_args = []
    k_center_macros = []


extensions = [
    Extension(
        "quant_cython_k_center",
        ["cython/k_center.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=k_center_compile_args,
        extra_link_args=k_center_link_args,
        define_macros=k_center_macros,
    ),
    Extension(
        "quant_cython_mdv",
        ["cython/mdv.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-fno-math-errno", "-fno-trapping-math"] + omp_cflags,
        extra_link_args=omp_lflags,
        define_macros=k_center_macros,
    ),
    Extension(
        "quant_cython_adv",
        ["cython/adv.pyx", "kmeans/interval_sum.cpp"],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=["-O3", "-march=native", "-std=c++14"] + omp_cflags,
        extra_link_args=omp_lflags,
        define_macros=k_center_macros,
        language="c++",
    ),
    Extension(
        "quant_cython_kmeans",
        [
            "cython/kmeans1d.pyx",
            "kmeans/kmeans_wilber.cpp",
            "kmeans/interval_sum.cpp",
        ],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=["-O3", "-march=native", "-std=c++14"],
        language="c++",
    ),
    Extension(
        "quant_cython_asq_wilber",
        [
            "cython/asq_wilber.pyx",
            "kmeans/interval_sum.cpp",
        ],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=["-O3", "-march=native", "-std=c++14", "-ffast-math"],
        language="c++",
    ),
    Extension(
        "quant_cython_pq",
        ["cython/pq.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-std=c++14"],
        language="c++",
    ),
]

setup(
    name="quantization-cython",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={"boundscheck": False, "wraparound": False, "cdivision": True},
    ),
)
