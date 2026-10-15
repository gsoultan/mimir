"""Measures float32 GEMM throughput of NumPy's BLAS, to compare with
mimir's BenchmarkSGEMM.

NumPy's wheels use OpenBLAS on Linux and Windows and Apple Accelerate on
macOS 14 or later; `python3 -c "import numpy; numpy.show_config()"` names
the one in use. Run with as many threads as mimir uses, for example:

    OPENBLAS_NUM_THREADS=4 python3 gemm.py
"""
import time

import numpy as np


def gflops(n):
    rng = np.random.default_rng(1)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    c = a @ b
    reps = max(3, int(2e10 / (2 * n**3)))
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            np.matmul(a, b, out=c)
        best = min(best, (time.perf_counter() - t) / reps)
    return 2 * n**3 / best / 1e9


for n in [64, 256, 512, 1024, 2048]:
    print(f"{n}\t{gflops(n):.1f} GFLOP/s")
