from __future__ import annotations

import argparse
import statistics
import time

import cupy as cp
import numpy as np

from ekfdmd import EKFDMD
from ekfdmd._cupy_experimental import CuPyEKFDMD


def _measure(function, repeat: int, synchronize=None) -> float:
    samples = []
    for _ in range(repeat):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        function()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _case(n: int, steps: int, repeat: int) -> tuple[float, float]:
    X = np.random.default_rng(n).standard_normal((n, steps))
    kwargs = dict(state_noise=1e-3, operator_noise=1e-5, measurement_noise=2e-3)

    cpu = EKFDMD(**kwargs).fit(X)
    gpu = CuPyEKFDMD(**kwargs).fit(X)
    np.testing.assert_allclose(gpu.operator, cpu.operator, rtol=1e-6, atol=1e-7)

    cpu_time = _measure(lambda: EKFDMD(**kwargs).fit(X), repeat)
    gpu_time = _measure(lambda: CuPyEKFDMD(**kwargs).fit(X), repeat, cp.cuda.Device().synchronize)
    return cpu_time, gpu_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NumPy/SciPy and CuPy EKF-DMD runtimes.")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    cp.cuda.runtime.getDeviceCount()
    CuPyEKFDMD().fit(np.random.default_rng(0).standard_normal((2, 4)))

    print("n,t,cpu_s,cupy_s,speedup")
    for n in (10, 20, 30, 40, 50):
        steps = 100
        cpu_time, gpu_time = _case(n, steps, args.repeat)
        print(f"{n},{steps},{cpu_time:.6f},{gpu_time:.6f},{cpu_time / gpu_time:.3f}")


if __name__ == "__main__":
    main()
