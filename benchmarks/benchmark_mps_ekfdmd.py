from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from ekfdmd import EKFDMD
from ekfdmd.mps import MPSEKFDMD


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


def _step_errors(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    difference = actual - expected
    numerator = np.linalg.norm(difference, axis=(1, 2))
    denominator = np.linalg.norm(expected, axis=(1, 2))
    return numerator / denominator


def _case(n: int, steps: int, repeat: int) -> tuple[float, ...]:
    X = np.random.default_rng(n).standard_normal((n, steps))
    kwargs = dict(state_noise=1e-3, operator_noise=1e-5, measurement_noise=2e-3)
    diagnostic_kwargs = {**kwargs, "history": "all"}

    cpu64 = EKFDMD(**kwargs).fit(X)
    cpu32 = MPSEKFDMD(**diagnostic_kwargs)._fit_device(X, torch.device("cpu"))
    mps = MPSEKFDMD(**diagnostic_kwargs).fit(X)
    mps_repeat = MPSEKFDMD(**kwargs).fit(X)

    mps_vs_cpu64 = _relative_error(mps.operator, cpu64.operator)
    cpu32_vs_cpu64 = _relative_error(cpu32.operator, cpu64.operator)
    mps_vs_cpu32 = _relative_error(mps.operator, cpu32.operator)
    mps_vs_cpu32_max = float(np.max(np.abs(mps.operator - cpu32.operator)))
    mps_repeat_rel = _relative_error(mps_repeat.operator, mps.operator)

    step_errors = _step_errors(mps.operators, cpu32.operators)
    first_update_rel = float(step_errors[1])
    max_step_rel = float(np.max(step_errors))
    over_one_percent = np.flatnonzero(step_errors > 1e-2)
    first_over_one_percent = int(over_one_percent[0]) if over_one_percent.size else -1

    cpu_time = _measure(lambda: EKFDMD(**kwargs).fit(X), repeat)
    mps_time = _measure(lambda: MPSEKFDMD(**kwargs).fit(X), repeat, torch.mps.synchronize)
    return (
        cpu_time,
        mps_time,
        mps_vs_cpu64,
        cpu32_vs_cpu64,
        mps_vs_cpu32,
        mps_vs_cpu32_max,
        first_update_rel,
        max_step_rel,
        first_over_one_percent,
        mps_repeat_rel,
    )


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
        "mps_vs_cpu32_rel,mps_vs_cpu32_max_abs,first_update_rel,max_step_rel,"
        "first_step_over_1pct,mps_repeat_rel"
    )
    for n in (10, 20, 30, 40, 50):
        steps = 100
        values = _case(n, steps, args.repeat)
        cpu_time, mps_time = values[:2]
        print(
            f"{n},{steps},{cpu_time:.6f},{mps_time:.6f},{cpu_time / mps_time:.3f},"
            f"{values[2]:.6e},{values[3]:.6e},{values[4]:.6e},{values[5]:.6e},"
            f"{values[6]:.6e},{values[7]:.6e},{int(values[8])},{values[9]:.6e}"
        )


if __name__ == "__main__":
    main()
