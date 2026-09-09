from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _worker(
    worker_id: int,
    n: int,
    steps: int,
    rank: int | None,
    fits_per_worker: int,
    device_id: int,
    barrier,
) -> tuple[str, float, float, str]:
    """Run independent FP64 EKFDMD fits in one CUDA process."""
    status = "ok"
    error = ""
    elapsed = 0.0
    pool_gib = 0.0

    try:
        import cupy as cp

        from ekfdmd.cupy import CuPyEKFDMD

        cp.cuda.Device(device_id).use()
        CuPyEKFDMD().fit(np.random.default_rng(worker_id).standard_normal((2, 4)))
        cp.cuda.Device().synchronize()
        X = np.random.default_rng(10_000 + worker_id).standard_normal((n, steps)).astype(np.float64)
    except Exception as exc:  # pragma: no cover - hardware dependent
        status = f"setup_{type(exc).__name__}"
        error = str(exc)

    try:
        barrier.wait(timeout=300)
    except Exception as exc:  # pragma: no cover - process failure path
        return f"barrier_{type(exc).__name__}", 0.0, 0.0, str(exc)

    if status != "ok":
        return status, elapsed, pool_gib, error

    try:
        start = time.perf_counter()
        for _ in range(fits_per_worker):
            model = CuPyEKFDMD(
                state_noise=1e-3,
                operator_noise=1e-5,
                measurement_noise=2e-3,
                rank=rank,
            ).fit(X)
            if model.operator.dtype != np.float64:
                raise RuntimeError(f"expected float64 output, got {model.operator.dtype}")
        cp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - start
        pool_gib = cp.get_default_memory_pool().total_bytes() / 2**30
    except cp.cuda.memory.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
        status = "oom"
        error = str(exc)
    except Exception as exc:  # pragma: no cover - hardware dependent
        status = f"error_{type(exc).__name__}"
        error = str(exc)

    return status, elapsed, pool_gib, error


def _gpu_memory_gib(device_id: int) -> tuple[float, float] | None:
    """Read global VRAM use through nvidia-smi without adding a Python dependency."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_id),
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        used_mib, total_mib = (float(value.strip()) for value in result.stdout.splitlines()[0].split(","))
        return used_mib / 1024, total_mib / 1024
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _run_level(
    concurrency: int,
    *,
    n: int,
    steps: int,
    rank: int | None,
    fits_per_worker: int,
    device_id: int,
    monitor_interval: float,
) -> tuple[str, float, float, float, float, float, float, list[str]]:
    ctx = mp.get_context("spawn")
    baseline_memory = _gpu_memory_gib(device_id)
    baseline_used = baseline_memory[0] if baseline_memory is not None else float("nan")
    peak_used = baseline_used

    with ctx.Manager() as manager:
        barrier = manager.Barrier(concurrency + 1)
        with ProcessPoolExecutor(max_workers=concurrency, mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    _worker,
                    worker_id,
                    n,
                    steps,
                    rank,
                    fits_per_worker,
                    device_id,
                    barrier,
                )
                for worker_id in range(concurrency)
            ]

            try:
                barrier.wait(timeout=300)
            except Exception as exc:
                return (
                    f"barrier_{type(exc).__name__}",
                    0.0,
                    0.0,
                    0.0,
                    peak_used,
                    float("nan"),
                    0.0,
                    [str(exc)],
                )

            start = time.perf_counter()
            while not all(future.done() for future in futures):
                memory = _gpu_memory_gib(device_id) if monitor_interval > 0 else None
                if memory is not None:
                    peak_used = max(peak_used, memory[0])
                if monitor_interval > 0:
                    time.sleep(monitor_interval)
            wall_time = time.perf_counter() - start

            memory = _gpu_memory_gib(device_id)
            if memory is not None:
                peak_used = max(peak_used, memory[0])

            results = [future.result() for future in futures]

    statuses = [result[0] for result in results]
    errors = [result[3] for result in results if result[3]]
    status = "ok" if all(value == "ok" for value in statuses) else next(value for value in statuses if value != "ok")
    worker_times = [result[1] for result in results if result[0] == "ok"]
    pool_total = sum(result[2] for result in results if result[0] == "ok")
    total_cases = concurrency * fits_per_worker if status == "ok" else len(worker_times) * fits_per_worker
    throughput = total_cases / wall_time if wall_time > 0 else 0.0
    median_worker_time = statistics.median(worker_times) if worker_times else 0.0
    max_worker_time = max(worker_times, default=0.0)
    peak_delta = peak_used - baseline_used if np.isfinite(peak_used) and np.isfinite(baseline_used) else float("nan")

    return (
        status,
        wall_time,
        throughput,
        median_worker_time,
        peak_used,
        peak_delta,
        pool_total,
        errors,
    )


def _parse_levels(value: str) -> list[int]:
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("concurrency levels must be positive integers")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure FP64 CuPy EKFDMD throughput under process concurrency.")
    parser.add_argument("--n", type=int, default=50, help="state dimension")
    parser.add_argument("--steps", type=int, default=100, help="number of snapshots")
    parser.add_argument("--rank", type=int, default=None, help="optional EKFDMD reduction rank")
    parser.add_argument("--fits-per-worker", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--concurrency",
        type=_parse_levels,
        default=_parse_levels("1,2,4,8,12,16,24,32,40"),
        help="comma-separated process counts",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.1,
        help="seconds between nvidia-smi VRAM samples; 0 disables sampling",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.rank is not None and not 1 <= args.rank <= args.n:
        parser.error("--rank must be between 1 and n")
    if args.fits_per_worker < 1:
        parser.error("--fits-per-worker must be at least 1")
    if args.monitor_interval < 0:
        parser.error("--monitor-interval must be non-negative")

    print(
        f"# FP64 CuPyEKFDMD: n={args.n}, steps={args.steps}, rank={args.rank}, "
        f"fits_per_worker={args.fits_per_worker}, device={args.device}"
    )
    print(
        "concurrency,status,wall_s,cases_per_s,throughput_vs_1x,median_worker_s,"
        "peak_vram_gib,peak_vram_delta_gib,sum_cupy_pool_gib"
    )

    baseline_throughput: float | None = None
    successful: list[tuple[int, float]] = []

    for concurrency in args.concurrency:
        result = _run_level(
            concurrency,
            n=args.n,
            steps=args.steps,
            rank=args.rank,
            fits_per_worker=args.fits_per_worker,
            device_id=args.device,
            monitor_interval=args.monitor_interval,
        )
        status, wall_time, throughput, median_worker, peak_vram, peak_delta, pool_total, errors = result
        if concurrency == 1 and status == "ok":
            baseline_throughput = throughput
        relative = throughput / baseline_throughput if baseline_throughput else float("nan")
        print(
            f"{concurrency},{status},{wall_time:.6f},{throughput:.6f},{relative:.3f},"
            f"{median_worker:.6f},{peak_vram:.3f},{peak_delta:.3f},{pool_total:.3f}"
        )

        if errors:
            for error in dict.fromkeys(errors):
                print(f"# {concurrency} workers: {error}", file=sys.stderr)

        if status == "ok":
            successful.append((concurrency, throughput))
        else:
            print(f"# stopped after first failed level: {concurrency} ({status})")
            break

    if successful:
        largest = successful[-1][0]
        best_concurrency, best_throughput = max(successful, key=lambda item: item[1])
        print(f"# largest_successful_concurrency={largest}")
        print(f"# best_throughput_concurrency={best_concurrency}")
        print(f"# best_throughput_cases_per_s={best_throughput:.6f}")


if __name__ == "__main__":
    main()
