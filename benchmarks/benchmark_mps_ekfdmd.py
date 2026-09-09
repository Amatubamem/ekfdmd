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


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _case(n: int, steps: int, repeat: int) -> tuple[float, float, float, float, float, float]:
    X = np.random.default_rng(n).standard_normal((n, steps))
    kwargs = dict(state_noise=1e-3, operator_noise=1e-5, measurement_noise=2e-3)

    cpu64 = EKFDMD(**kwargs).fit(X)
    cpu32 = MPSEKFDMD(**kwargs)._fit_device(X, torch.device("cpu"))
    mps = MPSEKFDMD(**kwargs).fit(X)

    mps_vs_cpu64 = _relative_error(mps.operator, cpu64.operator)
    cpu32_vs_cpu64 = _relative_error(cpu32.operator, cpu64.operator)
    mps_vs_cpu32 = _relative_error(mps.operator, cpu32.operator)
    mps_vs_cpu32_max = float(np.max(np.abs(mps.operator - cpu32.operator)))

    cpu_time = _measure(lambda: EKFDMD(**kwargs).fit(X), repeat)
    mps_time = _measure(lambda: MPSEKFDMD(**kwargs).fit(X), repeat, torch.mps.synchronize)
    return cpu_time, mps_time, mps_vs_cpu64, cpu32_vs_cpu64, mps_vs_cpu32, mps_vs_cpu32_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NumPy/SciPy and Apple MPS EKF-DMD runtimes.")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not torch.backends.mps.is_available():
        parser.error("Apple MPS is not available")

    MPSEKFDMD().fit(np.random.default_rng(0).standard_normal((2, 4)))

    print(
        "n,t,cpu_s,mps_s,speedup,mps_vs_cpu64_rel,cpu32_vs_cpu64_rel,"
        "mps_vs_cpu32_rel,mps_vs_cpu32_max_abs"
    )
    for n in (10, 20, 30, 40, 50):
        steps = 100
        values = _case(n, steps, args.repeat)
        cpu_time, mps_time, mps64, cpu3264, mps32, mps32max = values
        print(
            f"{n},{steps},{cpu_time:.6f},{mps_time:.6f},{cpu_time / mps_time:.3f},"
            f"{mps64:.6e},{cpu3264:.6e},{mps32:.6e},{mps32max:.6e}"
        )


if __name__ == "__main__":
    main()
