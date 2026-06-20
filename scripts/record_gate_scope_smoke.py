#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_torch_stub_if_needed() -> None:
    if importlib.util.find_spec("torch") is not None:
        return
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(Stream=object)
    sys.modules["torch"] = torch


def _record(
    *,
    task_id: str,
    backend: str,
    generator: str = "philox4x32_10",
    distribution: str = "raw32",
    validation_status: str = "pass",
    comparison_key: str | None = "candidate",
    formal_result: bool = True,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "backend": backend,
        "gate_backend": backend,
        "api_surface": "gate" if task_id.startswith("G") else "flagrand_public_api",
        "generator": generator,
        "distribution": distribution,
        "validation": {"status": validation_status, "checks": {}},
        "comparison_key": comparison_key,
        "formal_result": formal_result,
        "audit_flags": [],
    }


def main() -> None:
    _install_torch_stub_if_needed()

    from contract_benchmark.records import apply_cross_record_gates

    records = [
        _record(task_id="G2_REPRODUCIBILITY", backend="flagrand_public", validation_status="fail", comparison_key=None, formal_result=False),
        _record(task_id="K0_DEVICE_RAW_OUTPUT", backend="flagrand_public_output"),
        _record(task_id="I3_FIRST_VS_STEADY", backend="flagrand_public_first", distribution="uniform_f32"),
        _record(task_id="F1_ADD_UNIFORM", backend="flagrand_fused_philox", distribution="uniform_add_consume"),
        _record(task_id="I1_GENERATOR_LIFECYCLE", backend="flagrand_lifecycle", distribution="lifecycle"),
    ]
    apply_cross_record_gates(records)

    by_backend = {str(record["backend"]): record for record in records}
    assert by_backend["flagrand_public_output"]["formal_result"] is False
    assert by_backend["flagrand_public_output"]["gate_backend"] == "flagrand_public"
    assert "sequence_semantics_gate_failed" in by_backend["flagrand_public_output"]["audit_flags"]

    assert by_backend["flagrand_public_first"]["formal_result"] is False
    assert by_backend["flagrand_public_first"]["gate_backend"] == "flagrand_public"
    assert "sequence_semantics_gate_failed" in by_backend["flagrand_public_first"]["audit_flags"]

    assert by_backend["flagrand_fused_philox"]["formal_result"] is True
    assert by_backend["flagrand_lifecycle"]["formal_result"] is True
    print("record gate scope smoke ok")


if __name__ == "__main__":
    main()
