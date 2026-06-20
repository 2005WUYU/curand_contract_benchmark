from __future__ import annotations

from typing import Any

from contract_benchmark.curand_library import library_load_report
from contract_benchmark.generator_registry import GENERATOR_INFOS


def capability_matrix() -> dict[str, Any]:
    matrix: dict[str, Any] = {
        "curand_host": library_load_report(),
        "generators": {},
        "device_api_extension": optional_device_extension_status(),
        "curanddx": {
            "available": False,
            "unsupported_reason": "cuRANDDx headers/build integration are not configured in this local repository.",
        },
    }
    for name, info in GENERATOR_INFOS.items():
        matrix["generators"][name] = {
            "kind": info.kind,
            "curand_host": info.supports_curand_host,
            "flagrand": info.supports_flagrand,
            "raw32": info.supports_raw32,
            "raw64_native": info.supports_raw64,
            "distributions_f32": info.supports_distributions_f32,
            "supports_seed": info.supports_seed,
            "supports_offset": info.supports_offset,
            "notes": info.notes,
        }
    return matrix


def optional_device_extension_status() -> dict[str, Any]:
    try:
        from contract_benchmark.optional_device_api import find_built_curand_device_extension
    except Exception as exc:
        return {"available": False, "unsupported_reason": f"optional loader import failed: {exc}"}
    module, reason = find_built_curand_device_extension()
    return {
        "available": module is not None,
        "unsupported_reason": reason if module is None else None,
    }
