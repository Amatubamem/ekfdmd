from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from ekfdmd import EKFDMD
from ekfdmd._mps_experimental import MPSEKFDMD


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


def _case(n: int, steps: int, repeat: int) -> tuple[float, float, float, float]:
    X = np.random.default_rng(n).standard_normal((n, steps))
    kwargs = dict(state_noise=1e-3, operator_noise=1e-5, measurement_noise=2e-3)

    cpu = EKFDMD(**kwargs).fit(X)
    mps = MPSEKFDMD(**kwargs).fit(X)
    difference = mps.operator - cpu.operator
    relative_frobenius_error = np.linalg.norm(difference) / np.linalg.norm(cpu.operator)
    max_absolute_error = np.max(np.abs(difference))

    cpu_time = _measure(lambda: EKFDMD(**kwargs).fit(X), repeat)
    mps_time = _measure(lambda: MPSEKFDMD(**kwargs).fit(X), repeat, torch.mps.synchronize)
    return cpu_time, mps_time, relative_frobenius_error, max_absolute_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NumPy/SciPy and Apple MPS EKF-DMD runtimes.")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not torch.backends.mps.is_available():
        parser.error("Apple MPS is not available")

    MPSEKFDMD().fit(np.random.default_rng(0).standard_normal((2, 4)))

    print("n,t,cpu_s,mps_s,speedup,relative_frobenius_error,max_absolute_error")
    for n in (10, 20, 30, 40, 50):
        steps = 100
        cpu_time, mps_time, relative_error, max_error = _case(n, steps, args.repeat)
        print(
            f"{n},{steps},{cpu_time:.6f},{mps_time:.6f},{cpu_time / mps_time:.3f},"
            f"{relative_error:.6e},{max_error:.6e}"
        )


if __name__ == "__main__":
    main()
