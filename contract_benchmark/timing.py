from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class SampleStats:
    median_us: float | None
    mean_us: float | None
    stdev_us: float | None
    min_us: float | None
    max_us: float | None
    p25_us: float | None
    p75_us: float | None


@dataclass
class TimingResult:
    timer: str
    raw_samples_us: list[float]
    wall_sync_samples_us: list[float]
    cpu_enqueue_samples_us: list[float]
    gpu: SampleStats
    wall: SampleStats
    cpu_enqueue: SampleStats

    def to_record(self) -> dict[str, Any]:
        data = {
            "timer": self.timer,
            "raw_samples_us": self.raw_samples_us,
            "wall_sync_samples_us": self.wall_sync_samples_us,
            "cpu_enqueue_samples_us": self.cpu_enqueue_samples_us,
        }
        for prefix, stats in (
            ("gpu", self.gpu),
            ("wall_sync", self.wall),
            ("cpu_enqueue", self.cpu_enqueue),
        ):
            stats_dict = asdict(stats)
            for key, value in stats_dict.items():
                data[f"{key.replace('_us', '')}_{prefix}_us"] = value
        data["median_gpu_us"] = self.gpu.median_us
        data["median_wall_sync_us"] = self.wall.median_us
        data["median_cpu_enqueue_us"] = self.cpu_enqueue.median_us
        return data


def summarize_us(samples: list[float]) -> SampleStats:
    if not samples:
        return SampleStats(None, None, None, None, None, None, None)
    values = sorted(float(x) for x in samples)
    n = len(values)
    p25 = values[int((n - 1) * 0.25)]
    p75 = values[int((n - 1) * 0.75)]
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return SampleStats(
        median_us=float(statistics.median(values)),
        mean_us=float(statistics.mean(values)),
        stdev_us=float(stdev),
        min_us=float(values[0]),
        max_us=float(values[-1]),
        p25_us=float(p25),
        p75_us=float(p75),
    )


def warmup(run_once: Callable[[], object], *, warmup_iters: int) -> None:
    for _ in range(warmup_iters):
        run_once()
    torch.cuda.synchronize()


def collect_cuda_event_and_wall_us(
    run_once: Callable[[], object],
    *,
    warmup_iters: int,
    repeats: int,
    stream: torch.cuda.Stream | None = None,
) -> TimingResult:
    if stream is None:
        stream = torch.cuda.current_stream()
    warmup(run_once, warmup_iters=warmup_iters)
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    enqueue_samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_t0 = time.perf_counter()
        start.record(stream)
        enqueue_t0 = time.perf_counter()
        run_once()
        enqueue_t1 = time.perf_counter()
        end.record(stream)
        end.synchronize()
        wall_t1 = time.perf_counter()
        gpu_samples.append(float(start.elapsed_time(end)) * 1000.0)
        wall_samples.append((wall_t1 - wall_t0) * 1_000_000.0)
        enqueue_samples.append((enqueue_t1 - enqueue_t0) * 1_000_000.0)
    return TimingResult(
        timer="cuda_event_same_stream_and_wall_sync",
        raw_samples_us=gpu_samples,
        wall_sync_samples_us=wall_samples,
        cpu_enqueue_samples_us=enqueue_samples,
        gpu=summarize_us(gpu_samples),
        wall=summarize_us(wall_samples),
        cpu_enqueue=summarize_us(enqueue_samples),
    )


def collect_wall_only_us(
    run_once: Callable[[], object],
    *,
    warmup_iters: int,
    repeats: int,
    sync_cuda: bool = True,
) -> TimingResult:
    for _ in range(warmup_iters):
        run_once()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
    wall_samples: list[float] = []
    enqueue_samples: list[float] = []
    for _ in range(repeats):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        enqueue_t0 = time.perf_counter()
        run_once()
        enqueue_t1 = time.perf_counter()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_t1 = time.perf_counter()
        wall_samples.append((wall_t1 - wall_t0) * 1_000_000.0)
        enqueue_samples.append((enqueue_t1 - enqueue_t0) * 1_000_000.0)
    return TimingResult(
        timer="wall_sync_only",
        raw_samples_us=[],
        wall_sync_samples_us=wall_samples,
        cpu_enqueue_samples_us=enqueue_samples,
        gpu=summarize_us([]),
        wall=summarize_us(wall_samples),
        cpu_enqueue=summarize_us(enqueue_samples),
    )


def audit_flags(timing: TimingResult) -> list[str]:
    flags: list[str] = []
    gpu_median = timing.gpu.median_us
    wall_median = timing.wall.median_us
    if gpu_median is not None and gpu_median < 20.0:
        flags.append("gpu_event_below_20us_resolution_sensitive")
    if gpu_median is not None and timing.gpu.stdev_us is not None and gpu_median > 0:
        if timing.gpu.stdev_us / gpu_median > 0.25:
            flags.append("gpu_event_high_variability")
    if gpu_median is not None and wall_median is not None and gpu_median > 0:
        if wall_median / gpu_median > 3.0:
            flags.append("host_overhead_dominates_wall_time")
    return flags


def formal_result_from_flags(flags: list[str]) -> bool:
    blocking = {
        "gpu_event_below_20us_resolution_sensitive",
        "gpu_event_high_variability",
    }
    return not any(flag in blocking for flag in flags)

