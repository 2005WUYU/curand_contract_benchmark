from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from contract_benchmark.curand_ctypes import library_load_report
from contract_benchmark.spec import TaskSpec


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    sizes: list[int]
    gate_n: int
    warmup: int
    repeats: int
    raw_generators: list[str]
    dist_generators: list[str]
    qrng_generators: list[str]
    poisson_lambdas: list[float]
    fused_ps: list[float]
    many_small_calls: int
    many_small_chunk_n: int


PROFILES = {
    "local_smoke": BenchmarkProfile(
        "local_smoke",
        sizes=[1024, 16384, 65536],
        gate_n=4096,
        warmup=1,
        repeats=3,
        raw_generators=["philox4x32_10", "xorwow", "mrg32k3a"],
        dist_generators=["philox4x32_10"],
        qrng_generators=["sobol32", "sobol64"],
        poisson_lambdas=[1.0, 10.0],
        fused_ps=[0.5],
        many_small_calls=8,
        many_small_chunk_n=1024,
    ),
    "local": BenchmarkProfile(
        "local",
        sizes=[1024, 4096, 65536, 1048576],
        gate_n=16384,
        warmup=3,
        repeats=10,
        raw_generators=["philox4x32_10", "xorwow", "mrg32k3a", "sobol32", "sobol64"],
        dist_generators=["philox4x32_10", "xorwow", "mrg32k3a"],
        qrng_generators=["sobol32", "scrambled_sobol32", "sobol64", "scrambled_sobol64"],
        poisson_lambdas=[0.1, 1.0, 10.0, 64.0],
        fused_ps=[0.1, 0.5, 0.9],
        many_small_calls=64,
        many_small_chunk_n=1024,
    ),
    "h20": BenchmarkProfile(
        "h20",
        sizes=[4096, 16384, 65536, 262144, 1048576, 4194304, 8388608],
        gate_n=65536,
        warmup=5,
        repeats=20,
        raw_generators=["philox4x32_10", "xorwow", "mrg32k3a", "mtgp32", "mt19937", "sobol32", "sobol64"],
        dist_generators=["philox4x32_10", "xorwow", "mrg32k3a"],
        qrng_generators=["sobol32", "scrambled_sobol32", "sobol64", "scrambled_sobol64"],
        poisson_lambdas=[0.1, 1.0, 4.0, 10.0, 32.0, 64.0, 256.0, 1024.0, 10000.0],
        fused_ps=[0.01, 0.1, 0.5, 0.9, 0.99],
        many_small_calls=128,
        many_small_chunk_n=1024,
    ),
}


@dataclass
class BenchmarkContext:
    repo_root: Path
    benchmark_root: Path
    profile: BenchmarkProfile
    specs: dict[str, TaskSpec]
    seed: int = 12345
    offset: int = 0
    device: torch.device = torch.device("cuda")


def collect_environment(profile_name: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    try:
        curand_report = library_load_report()
    except Exception as exc:
        curand_report = {"available": False, "error": str(exc)}
    env = {
        "profile": profile_name,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime_from_torch": torch.version.cuda,
        "curand": curand_report,
        "time_unix": time.time(),
    }
    if cuda_available:
        env.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_count": torch.cuda.device_count(),
            }
        )
        env["nvidia_smi"] = _nvidia_smi_snapshot()
    try:
        import triton

        env["triton_version"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:
        env["triton_version_error"] = str(exc)
    env["git"] = _git_info()
    launcher_git_commit = os.environ.get("CURAND_CONTRACT_GIT_SHA")
    if launcher_git_commit:
        env["launcher_git_commit"] = launcher_git_commit
    return env


def _git_info() -> dict[str, Any]:
    launcher_git_commit = os.environ.get("CURAND_CONTRACT_GIT_SHA")
    try:
        root = Path(__file__).resolve().parents[1]
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return {"commit": sha, "dirty": bool(status.strip()), "status_short": status.splitlines()[:20], "source": "git"}
    except Exception as exc:
        info: dict[str, Any] = {"error": str(exc)}
        if launcher_git_commit:
            info.update({"commit": launcher_git_commit, "dirty": None, "status_short": [], "source": "CURAND_CONTRACT_GIT_SHA"})
        return info


def _nvidia_smi_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    query_fields = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "pstate",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.applications.graphics",
        "clocks.applications.memory",
    ]
    query_cmd = ["nvidia-smi", f"--query-gpu={','.join(query_fields)}", "--format=csv,noheader,nounits"]
    try:
        query_output = subprocess.check_output(query_cmd, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        snapshot["query_fields"] = query_fields
        snapshot["query_rows"] = [line.split(", ") for line in query_output.splitlines() if line.strip()]
    except Exception as exc:
        snapshot["query_error"] = str(exc)

    try:
        q_output = subprocess.check_output(["nvidia-smi", "-q"], text=True, stderr=subprocess.STDOUT, timeout=10)
        max_chars = int(os.environ.get("CURAND_CONTRACT_NVIDIA_SMI_Q_MAX_CHARS", "50000"))
        snapshot["q_text"] = q_output[:max_chars]
        snapshot["q_text_truncated"] = len(q_output) > max_chars
    except Exception as exc:
        snapshot["q_error"] = str(exc)
    return snapshot
