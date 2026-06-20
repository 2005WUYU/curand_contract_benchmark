from __future__ import annotations

from typing import Any

import torch


def validation_pass(checks: dict[str, Any]) -> dict[str, Any]:
    boolean_checks = [v for v in checks.values() if isinstance(v, bool)]
    status = "pass" if boolean_checks and all(boolean_checks) else "fail"
    return {"status": status, "checks": checks}


def validation_error(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "fail",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def unsupported(reason: str, checks: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "unsupported_reason": reason,
        "checks": checks or {},
    }


def validate_raw_tensor(out: torch.Tensor, *, dtype: torch.dtype, n: int) -> dict[str, Any]:
    sample = _sample_cpu(out)
    checks = {
        "device_type": out.device.type == "cuda",
        "dtype": str(out.dtype) == str(dtype),
        "shape_numel": out.numel() == n,
        "contiguous": out.is_contiguous(),
        "sample_nonempty": sample.numel() > 0,
    }
    if sample.numel() > 1:
        checks["sample_not_all_same"] = bool((sample != sample[0]).any().item())
    return validation_pass(checks)


def validate_uniform(out: torch.Tensor, *, n: int, low_open: bool | None = None) -> dict[str, Any]:
    sample = _sample_cpu(out.float())
    checks: dict[str, Any] = {
        "device_type": out.device.type == "cuda",
        "dtype": str(out.dtype) in ("torch.float32", "torch.float64"),
        "shape_numel": out.numel() == n,
        "all_finite_sample": bool(torch.isfinite(sample).all().item()) if sample.numel() else False,
    }
    if sample.numel():
        min_v = float(sample.min().item())
        max_v = float(sample.max().item())
        mean_v = float(sample.mean().item())
        checks.update(
            {
                "sample_min": min_v,
                "sample_max": max_v,
                "sample_mean": mean_v,
                "range_0_1_sample": min_v >= 0.0 and max_v <= 1.0,
                "mean_rough": 0.35 <= mean_v <= 0.65,
            }
        )
        if low_open is True:
            checks["low_endpoint_open_sample"] = min_v > 0.0
    return validation_pass(checks)


def validate_normal(out: torch.Tensor, *, n: int, mean: float, stddev: float) -> dict[str, Any]:
    sample = _sample_cpu(out.float())
    checks: dict[str, Any] = {
        "device_type": out.device.type == "cuda",
        "dtype": str(out.dtype) in ("torch.float32", "torch.float64"),
        "shape_numel": out.numel() == n,
        "all_finite_sample": bool(torch.isfinite(sample).all().item()) if sample.numel() else False,
    }
    if sample.numel() > 8:
        sample_mean = float(sample.mean().item())
        sample_std = float(sample.std(unbiased=False).item())
        checks.update(
            {
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "mean_rough": abs(sample_mean - mean) <= max(0.20, 0.25 * abs(stddev)),
                "std_rough": 0.55 * stddev <= sample_std <= 1.55 * stddev,
            }
        )
    return validation_pass(checks)


def validate_lognormal(out: torch.Tensor, *, n: int) -> dict[str, Any]:
    sample = _sample_cpu(out.float())
    checks: dict[str, Any] = {
        "device_type": out.device.type == "cuda",
        "dtype": str(out.dtype) in ("torch.float32", "torch.float64"),
        "shape_numel": out.numel() == n,
        "all_finite_sample": bool(torch.isfinite(sample).all().item()) if sample.numel() else False,
    }
    if sample.numel():
        checks.update(
            {
                "sample_min": float(sample.min().item()),
                "sample_mean": float(sample.mean().item()),
                "positive_sample": bool((sample > 0).all().item()),
            }
        )
    return validation_pass(checks)


def validate_poisson(out: torch.Tensor, *, n: int, lambda_val: float) -> dict[str, Any]:
    sample = _sample_cpu(out.to(torch.float32))
    checks: dict[str, Any] = {
        "device_type": out.device.type == "cuda",
        "dtype": str(out.dtype) in ("torch.int32", "torch.int64"),
        "shape_numel": out.numel() == n,
    }
    if sample.numel():
        min_v = float(sample.min().item())
        mean_v = float(sample.mean().item())
        var_v = float(sample.var(unbiased=False).item()) if sample.numel() > 1 else 0.0
        checks.update(
            {
                "sample_min": min_v,
                "sample_mean": mean_v,
                "sample_variance": var_v,
                "nonnegative": min_v >= 0,
                "mean_rough": abs(mean_v - lambda_val) <= max(3.0, 0.50 * lambda_val),
                "variance_rough": abs(var_v - lambda_val) <= max(5.0, 0.75 * lambda_val),
            }
        )
    return validation_pass(checks)


def validate_mask(mask: torch.Tensor, *, n: int, p: float) -> dict[str, Any]:
    sample = _sample_cpu(mask.to(torch.float32))
    checks: dict[str, Any] = {
        "device_type": mask.device.type == "cuda",
        "dtype": str(mask.dtype) == "torch.uint8",
        "shape_numel": mask.numel() == n,
    }
    if sample.numel():
        ratio = float(sample.mean().item())
        checks.update(
            {
                "sample_keep_ratio": ratio,
                "ratio_rough": abs(ratio - p) <= max(0.10, 0.35 * min(p, 1.0 - p)),
            }
        )
    return validation_pass(checks)


def validate_finite_output(out: torch.Tensor, *, n: int) -> dict[str, Any]:
    sample = _sample_cpu(out.float())
    checks = {
        "device_type": out.device.type == "cuda",
        "shape_numel": out.numel() == n,
        "all_finite_sample": bool(torch.isfinite(sample).all().item()) if sample.numel() else False,
    }
    return validation_pass(checks)


def tensors_equal(a: torch.Tensor, b: torch.Tensor, *, max_items: int = 4096) -> bool:
    aa = _sample_cpu(a, max_items=max_items)
    bb = _sample_cpu(b, max_items=max_items)
    return bool(torch.equal(aa, bb))


def _sample_cpu(out: torch.Tensor, *, max_items: int = 4096) -> torch.Tensor:
    if out.numel() == 0:
        return out.detach().cpu()
    count = min(out.numel(), max_items)
    return out.detach().flatten()[:count].cpu()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return True
