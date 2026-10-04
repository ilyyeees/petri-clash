"""Small, explicit CPU runtime controls shared by the app and benchmark.

Nothing is configured at import time.  A single intra-op thread is a conservative
default for these small grids; use ``--cpu-threads`` or ``PETRI_CPU_THREADS`` to
measure a different setting on your machine.  Reproducibility applies to the
same PyTorch version, device, model, parameters and sequence of actions.
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import time
from collections.abc import Callable

import numpy as np
import torch


DEFAULT_CPU_THREADS = 1
CPU_THREADS_ENV = "PETRI_CPU_THREADS"


def resolve_cpu_threads(cpu_threads: int | None = None) -> int:
    """Resolve explicit argument, environment, then the small-grid default."""
    value = cpu_threads if cpu_threads is not None else os.getenv(CPU_THREADS_ENV, DEFAULT_CPU_THREADS)
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CPU threads must be a positive integer") from exc
    if isinstance(value, bool) or count < 1 or (isinstance(value, float) and value != count):
        raise ValueError("CPU threads must be a positive integer")
    return count


def seed_all(seed: int = 0) -> int:
    """Seed simulation and placement RNGs without enabling a GPU backend."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def runtime_info(seed: int | None = None) -> dict:
    """JSON-safe environment information; intentionally excludes host paths."""
    return {
        "device": "cpu",
        "seed": seed,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "logical_cpu_count": os.cpu_count(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "platform": platform.system(),
        "machine": platform.machine(),
    }


def configure_runtime(
    seed: int = 0, cpu_threads: int | None = None, deterministic: bool = True
) -> dict:
    """Configure before model construction/stepping and return actual settings.

    This deliberately does not change inter-op threads: PyTorch only permits
    that setting before its first parallel operation, making repeated app/test
    setup unsafe.  No compiler, accelerator, or trainer setting is enabled.
    """
    count = resolve_cpu_threads(cpu_threads)
    # Validate first, so invalid input does not leave half-applied settings.
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a boolean")
    torch.set_num_threads(count)
    torch.use_deterministic_algorithms(deterministic)
    seed_all(seed)
    return runtime_info(seed)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = (len(sorted_values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def benchmark_steps(step: Callable[[], object], *, warmup: int = 20, steps: int = 100) -> dict:
    """Time CPU simulation steps only, excluding warmup, rendering and waits.

    ``step`` owns its evolving state.  Warmup advances the same simulation;
    each recorded sample is exactly one subsequent call in inference mode.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            step()
        for _ in range(steps):
            started = time.perf_counter_ns()
            step()
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    samples.sort()
    total_ms = sum(samples)
    return {
        "warmup_steps": warmup,
        "measured_steps": steps,
        "step_ms": {
            "min": samples[0],
            "p50": _percentile(samples, 0.50),
            "p95": _percentile(samples, 0.95),
            "max": samples[-1],
            "mean": statistics.fmean(samples),
        },
        "total_measured_ms": total_ms,
        "steps_per_second": steps * 1000.0 / total_ms if total_ms else None,
    }
